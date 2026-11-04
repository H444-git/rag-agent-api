"""
针对提交 3e1c713 中关键缺陷的回归测试。

覆盖的缺陷：
1. LLM 工具调用参数 JSON 解析失败时 /chat 返回 500（已修复为返回错误提示）。
2. call_llm 调用 requests.post 未设置 timeout，导致请求可无限挂起（已修复）。
3. knowledge 列表与 FAISS index 在 /chat 与 /upload 并发下非线程安全（已加锁）。
4. 知识库为空时 np.array([]).shape[1] 触发 IndexError 导致启动崩溃（已修复）。
"""

import io
import sys
import threading
import importlib
from unittest.mock import patch, MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# 在导入 agent 之前，mock 掉需要联网下载模型的 text2vec 以及需要 API key 的 tavily，
# 保证测试环境无需联网即可导入。
# ---------------------------------------------------------------------------
text2vec_mock = MagicMock()


def _fake_encode(texts):
    """模拟 SentenceModel.encode：字符串返回单个向量，列表返回对应数量的向量。"""
    dim = 768
    if isinstance(texts, str):
        return np.array([0.1] * dim, dtype="float32")
    return np.array([[0.1] * dim for _ in texts], dtype="float32")


text2vec_mock.SentenceModel.return_value.encode.side_effect = _fake_encode
sys.modules["text2vec"] = text2vec_mock

tavily_mock = MagicMock()
sys.modules["tavily"] = tavily_mock

import agent  # noqa: E402


def test_rag_chat_handles_malformed_tool_call_args():
    """模型返回的工具调用参数不是合法 JSON 时，rag_chat 不应抛异常导致 500。"""
    bad_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": "{query: 重庆天气,}",  # 非法 JSON
                            }
                        }
                    ],
                }
            }
        ]
    }

    with patch.object(agent, "search_mostk", return_value=[]), patch.object(
        agent.model, "encode", side_effect=_fake_encode
    ), patch.object(agent, "call_llm", return_value=bad_response):
        result = agent.rag_chat("今天重庆天气怎么样", [])

    assert isinstance(result, str)
    assert "解析失败" in result


def test_call_llm_passes_timeout():
    """call_llm 必须向 requests.post 传递 timeout，避免请求无限挂起。"""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }

    with patch.object(agent.requests, "post", return_value=mock_resp) as mock_post:
        agent.call_llm("你好", "search")

    assert mock_post.called
    _, kwargs = mock_post.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"] == 60


def test_concurrent_add_and_search_no_crash():
    """并发执行 add_knowledge 与 search_mostk 不应崩溃或损坏索引。"""
    dummy_vec = np.array([0.1] * agent.dim, dtype="float32")
    errors = []

    def _add():
        try:
            for _ in range(15):
                agent.add_knowledge(["并发测试块"])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def _search():
        try:
            for _ in range(15):
                agent.search_mostk(dummy_vec, K=4)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_add) for _ in range(3)] + [
        threading.Thread(target=_search) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发访问出现异常: {errors}"


def test_empty_knowledge_does_not_crash_at_import():
    """知识库文件为空时，模块导入不应因 np.array([]).shape[1] 抛出 IndexError。"""
    real_open = open

    def _fake_open(path, *args, **kwargs):
        if isinstance(path, str) and path.endswith("知识库简易版.txt"):
            return io.StringIO("")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=_fake_open):
        importlib.reload(agent)

    # 空知识库下索引仍能正常创建，维度合法
    assert agent.index.ntotal == 0
    assert agent.dim > 0
