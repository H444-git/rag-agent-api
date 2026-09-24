"""针对本次修复的回归测试。

每个测试都构造了具体的触发场景：修复存在则通过，修复被回退则失败
（抛 IndexError / JSONDecodeError / AttributeError）。
"""
import builtins
import importlib
from unittest.mock import MagicMock, patch

import agent


def test_empty_knowledge_does_not_crash_on_import():
    """修复 A：知识库文件缺失/为空时，模块导入不应抛 IndexError。

    触发场景：知识库简易版.txt 不存在 -> load_knowledge 返回 [] ->
    np.array([]) 是 1 维数组 -> 原代码 shape[1] 抛 IndexError 启动崩溃。
    """
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if "知识库简易版.txt" in str(path):
            raise FileNotFoundError(str(path))
        return real_open(path, *args, **kwargs)

    try:
        with patch("builtins.open", side_effect=fake_open):
            importlib.reload(agent)  # 修复前会在此抛 IndexError
        assert agent.knowledge == [], "知识库应为空"
        assert agent.index is not None, "空知识库也应创建索引对象"
        assert agent.index.ntotal == 0, "空知识库索引不应包含向量"
    finally:
        # 恢复正常模块状态，避免影响其它测试
        importlib.reload(agent)


def test_call_llm_returns_error_on_non_200():
    """修复 B：DeepSeek 返回非 200 且响应体非 JSON 时不应崩溃。

    触发场景：502 网关错误返回 HTML -> response.json() 抛异常 ->
    原代码未做状态码检查，/chat 接口 500 崩溃。
    """
    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.text = "<html>Bad Gateway</html>"
    mock_response.json.side_effect = ValueError("not json")

    with patch.object(agent.requests, "post", return_value=mock_response):
        # mode="chat" 且 tools 非空 -> use_stream=False -> 走 response.json() 分支
        result = agent.call_llm("hi", "chat", tools=[{"type": "function", "function": {"name": "x"}}])

    assert isinstance(result, dict), "应返回错误 dict 而非抛异常"
    assert "error" in result
    # rag_chat 会用 result["error"] 作为返回字符串，确认是字符串
    assert isinstance(result["error"], str)


def test_add_knowledge_and_search_mostk_acquire_faiss_lock():
    """修复 C：写入与检索必须互斥访问 FAISS 索引。

    触发场景：/upload（add_knowledge 写 index）与 /chat（search_mostk 读 index）
    并发 -> FAISS 不支持并发读写 -> 可能段错误。修复用 _faiss_lock 串行化。
    """
    assert hasattr(agent, "_faiss_lock"), "修复 C 应引入 _faiss_lock"

    acquired = []
    real_lock = agent._faiss_lock

    class RecordingLock:
        def __enter__(self):
            acquired.append("acquire")
            return self

        def __exit__(self, *exc):
            acquired.append("release")
            return False

    agent._faiss_lock = RecordingLock()
    try:
        agent.add_knowledge(["新增知识块内容"])
        assert "acquire" in acquired, "add_knowledge 应持锁"

        acquired.clear()
        query_vec = agent.model.encode("查询问题")
        agent.search_mostk(query_vec)
        assert "acquire" in acquired, "search_mostk 应持锁"
    finally:
        agent._faiss_lock = real_lock
