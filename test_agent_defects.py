"""
针对 agent.py 中高影响缺陷的回归测试。

由于 numpy / faiss / text2vec 等重型依赖未安装，这里通过 sys.modules
注入轻量 mock 后再导入 agent，聚焦验证缺陷行为本身。
"""
import sys
import json
import threading
import time
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 依赖 mock：在 import agent 之前注入
# ---------------------------------------------------------------------------

class _FakeArray:
    """最小化的 numpy 数组替身，支持 .astype / .shape / 迭代。"""

    def __init__(self, data):
        self._data = data

    @property
    def shape(self):
        d = self._data
        if isinstance(d, list) and d and isinstance(d[0], list):
            return (len(d), len(d[0]))
        if isinstance(d, list):
            return (len(d),)
        return ()

    def astype(self, _dtype):
        return self

    def __iter__(self):
        return iter(self._data)


def _install_mocks():
    # numpy
    np_mock = MagicMock()
    np_mock.array.side_effect = lambda data: _FakeArray(data)
    sys.modules["numpy"] = np_mock

    # faiss
    faiss_mock = MagicMock()
    faiss_mock.normalize_L2 = MagicMock()
    sys.modules["faiss"] = faiss_mock

    # fitz (pymupdf)
    sys.modules["fitz"] = MagicMock()

    # text2vec
    text2vec_mock = MagicMock()
    model_mock = MagicMock()
    # encode 返回与输入等长的二维向量列表（每行 8 维，便于测试）
    def _encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]
    model_mock.encode.side_effect = _encode
    text2vec_mock.SentenceModel.return_value = model_mock
    sys.modules["text2vec"] = text2vec_mock

    # langchain_text_splitters
    lc_mock = MagicMock()
    splitter_mock = MagicMock()
    splitter_mock.split_text.side_effect = lambda text: [text] if text else []
    lc_mock.RecursiveCharacterTextSplitter.return_value = splitter_mock
    sys.modules["langchain_text_splitters"] = lc_mock

    # tavily
    sys.modules["tavily"] = MagicMock()

    # docx
    sys.modules["docx"] = MagicMock()

    # dotenv
    dotenv_mock = MagicMock()
    dotenv_mock.load_dotenv = MagicMock()
    sys.modules["dotenv"] = dotenv_mock

    # requests
    sys.modules["requests"] = MagicMock()

    # colorlog（logger_utils 依赖）
    import logging

    colorlog_mock = MagicMock()
    colorlog_mock.ColoredFormatter = MagicMock()
    # 返回真实的 StreamHandler，避免其 .level 是 MagicMock 导致日志比较失败
    colorlog_mock.StreamHandler = logging.StreamHandler
    sys.modules["colorlog"] = colorlog_mock


_install_mocks()

import agent  # noqa: E402


# ---------------------------------------------------------------------------
# 测试：add_knowledge 并发安全（数据完整性）
# ---------------------------------------------------------------------------

def test_add_knowledge_empty_chunks_no_side_effects():
    """空块列表不应触发 encode/index.add，直接返回 (0, 当前总数)。"""
    original_len = len(agent.knowledge)
    with patch.object(agent.model, "encode") as enc, patch.object(
        agent.index, "add"
    ) as idx_add:
        added, total = agent.add_knowledge([])
    assert added == 0
    assert total == original_len
    enc.assert_not_called()
    idx_add.assert_not_called()


def test_add_knowledge_and_search_stay_in_sync():
    """
    回归：并发上传时 knowledge 列表与 FAISS index 可能错位。
    通过让 index.add 内部短暂阻塞，验证另一个线程无法在
    knowledge.extend 与 index.add 之间插入操作，保证顺序一致。
    """
    call_log = []  # 记录 (thread_id, operation)

    real_extend = agent.knowledge.extend
    real_index_add = agent.index.add

    def slow_index_add(vecs):
        call_log.append(("add", "start"))
        time.sleep(0.05)
        call_log.append(("add", "end"))

    def tracked_extend(chunks):
        call_log.append(("extend", len(chunks)))
        real_extend(chunks)

    with patch.object(agent.index, "add", side_effect=slow_index_add), patch.object(
        agent.knowledge, "extend", side_effect=tracked_extend
    ):
        t1 = threading.Thread(target=agent.add_knowledge, args=(["chunk-A1", "chunk-A2"],))
        t2 = threading.Thread(target=agent.add_knowledge, args=(["chunk-B1"],))
        t1.start()
        # 让 t1 先进入锁
        time.sleep(0.01)
        t2.start()
        t1.join()
        t2.join()

    # 验证：extend 与 add 必须成对出现，且不会交叉
    # 期望顺序: extend(A), add(start), add(end), extend(B), add(start), add(end)
    # 关键断言：不会出现 extend(B) 夹在 extend(A) 与 add(A) 之间
    extend_indices = [i for i, (op, _) in enumerate(call_log) if op == "extend"]
    add_start_indices = [i for i, (op, _) in enumerate(call_log) if op == "add" and _ == "start"]
    # 每个 extend 后面必须紧跟对应的 add(start)，中间不能有另一个 extend
    for ext_idx, add_idx in zip(extend_indices, add_start_indices):
        assert ext_idx < add_idx, "extend 必须在对应 add 之前"
        between = call_log[ext_idx + 1 : add_idx]
        assert not any(op == "extend" for op, _ in between), (
            "并发上传导致 knowledge.extend 与 index.add 错位"
        )

    # 最终 knowledge 与 index 的增量数量必须一致
    assert agent.knowledge.count("chunk-A1") == 1
    assert agent.knowledge.count("chunk-A2") == 1
    assert agent.knowledge.count("chunk-B1") == 1


# ---------------------------------------------------------------------------
# 测试：call_llm 请求超时（资源管理）
# ---------------------------------------------------------------------------

def test_call_llm_passes_timeout():
    """requests.post 必须带 timeout 参数，否则 API 挂起会耗尽线程池。"""
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    with patch.object(agent.requests, "post", return_value=fake_resp) as post:
        agent.call_llm("hello", "search")
    _, kwargs = post.call_args
    assert "timeout" in kwargs, "requests.post 缺少 timeout 参数"
    assert kwargs["timeout"] is not None


def test_call_llm_timeout_returns_error():
    """API 超时时应返回错误字典，而非抛出异常导致请求 500。"""
    import requests as _requests

    with patch.object(
        agent.requests, "post", side_effect=_requests.exceptions.Timeout()
    ):
        result = agent.call_llm("hello", "search")
    assert isinstance(result, dict)
    assert "error" in result


def test_call_llm_http_error_returns_error():
    """HTTP 错误（4xx/5xx）应被捕获并返回错误字典。"""
    import requests as _requests

    with patch.object(
        agent.requests, "post", side_effect=_requests.exceptions.HTTPError("500")
    ):
        result = agent.call_llm("hello", "search")
    assert isinstance(result, dict)
    assert "error" in result


# ---------------------------------------------------------------------------
# 测试：rag_chat 工具调用参数解析（崩溃风险）
# ---------------------------------------------------------------------------

def _make_tool_response(name, arguments):
    """构造一个带 tool_calls 的 LLM 响应。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            }
                        }
                    ],
                }
            }
        ]
    }


def test_rag_chat_malformed_tool_arguments_no_crash():
    """LLM 返回非法 JSON 参数时，应返回友好错误，不应抛出 JSONDecodeError。"""
    bad_resp = _make_tool_response("search_web", "{not valid json")
    # 知识库非空才能进入检索流程
    with patch.object(agent, "knowledge", ["some knowledge"]), patch.object(
        agent, "call_llm", return_value=bad_resp
    ):
        # 不应抛出异常
        result = agent.rag_chat("搜索北京天气", [])
    assert isinstance(result, str)
    assert "错误" in result or "参数" in result


def test_rag_chat_search_web_missing_query_no_crash():
    """search_web 缺少 query 字段时，应返回友好错误，不应抛出 KeyError。"""
    resp = _make_tool_response("search_web", json.dumps({}))  # 缺 query
    with patch.object(agent, "knowledge", ["some knowledge"]), patch.object(
        agent, "call_llm", return_value=resp
    ):
        result = agent.rag_chat("搜索一下", [])
    assert isinstance(result, str)
    assert "query" in result or "缺少" in result


def test_rag_chat_calculate_missing_expression_no_crash():
    """calculate 缺少 expression 字段时，应返回友好错误，不应抛出 KeyError。"""
    resp = _make_tool_response("calculate", json.dumps({}))
    with patch.object(agent, "knowledge", ["some knowledge"]), patch.object(
        agent, "call_llm", return_value=resp
    ):
        result = agent.rag_chat("算一下", [])
    assert isinstance(result, str)
    assert "expression" in result or "缺少" in result


def test_rag_chat_empty_choices_no_crash():
    """choices 为空列表时不应抛出 IndexError。"""
    with patch.object(agent, "knowledge", ["some knowledge"]), patch.object(
        agent, "call_llm", return_value={"choices": []}
    ):
        # 不应抛出 IndexError；循环结束后返回最大轮次提示
        result = agent.rag_chat("你好", [])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 测试：知识库为空时不崩溃（启动稳定性）
# ---------------------------------------------------------------------------

def test_search_mostk_empty_index_no_crash():
    """index.ntotal == 0 时 search 不应崩溃，应返回空列表。"""
    # 构造空索引：search 返回 indices=-1
    agent.index.search.return_value = (
        [[-1.0]],
        [[-1]],
    )
    matched = agent.search_mostk([0.0] * 8, K=4)
    assert matched == []
