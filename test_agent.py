"""回归测试：覆盖已修复的关键缺陷。

通过 mock 掉 text2vec / faiss / tavily 等重依赖，使测试可在无模型下载的环境下运行。
"""
import sys
import types
import json
import threading
import time

import numpy as np
import pytest
import requests
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 在导入 agent 之前，安装重依赖的替身模块
# ---------------------------------------------------------------------------
def _install_dependency_mocks():
    # text2vec
    text2vec = types.ModuleType("text2vec")

    class _FakeSentenceModel:
        def __init__(self, *args, **kwargs):
            self._rng = np.random.default_rng(42)

        def encode(self, texts):
            # 与真实 text2vec 行为一致：字符串返回 1D 向量，列表返回 2D
            if isinstance(texts, str):
                return self._rng.random((8,), dtype="float32")
            n = len(texts)
            return self._rng.random((n, 8), dtype="float32")

    text2vec.SentenceModel = _FakeSentenceModel
    sys.modules["text2vec"] = text2vec

    # faiss
    faiss = types.ModuleType("faiss")

    class _FakeIndex:
        def __init__(self, dim):
            self.dim = dim
            self.ntotal = 0
            self._vectors = np.empty((0, dim), dtype="float32")

        def add(self, vecs):
            vecs = np.asarray(vecs, dtype="float32")
            if self._vectors.size == 0:
                self._vectors = vecs
            else:
                self._vectors = np.vstack([self._vectors, vecs])
            self.ntotal = self._vectors.shape[0]

        def search(self, query, k):
            if self.ntotal == 0:
                return (
                    np.array([[-1.0]], dtype="float32"),
                    np.array([[-1]], dtype="int64"),
                )
            query = np.asarray(query, dtype="float32")
            sims = query @ self._vectors.T
            order = np.argsort(-sims[0])[:k]
            return sims[:, order].astype("float32"), order.reshape(1, -1).astype("int64")

    faiss.IndexFlatIP = _FakeIndex
    faiss.normalize_L2 = lambda x: None
    sys.modules["faiss"] = faiss

    # langchain_text_splitters
    lcts = types.ModuleType("langchain_text_splitters")

    class _FakeSplitter:
        def __init__(self, **kwargs):
            pass

        def split_text(self, text):
            return [text] if text else []

    lcts.RecursiveCharacterTextSplitter = _FakeSplitter
    sys.modules["langchain_text_splitters"] = lcts

    # tavily
    tavily = types.ModuleType("tavily")
    tavily.TavilyClient = MagicMock()
    sys.modules["tavily"] = tavily

    # fitz (pymupdf)
    fitz = types.ModuleType("fitz")
    fitz.open = MagicMock()
    sys.modules["fitz"] = fitz

    # docx
    docx_mod = types.ModuleType("docx")
    docx_mod.Document = MagicMock()
    sys.modules["docx"] = docx_mod

    # dotenv
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv


_install_dependency_mocks()

import agent  # noqa: E402


# ---------------------------------------------------------------------------
# Bug 1：知识库为空时启动不再崩溃（IndexError: shape[1]）
# ---------------------------------------------------------------------------
def test_empty_knowledge_base_does_not_crash_at_startup():
    """当知识库文件缺失/为空时，模块加载不应抛出 IndexError。"""
    import importlib
    import io

    real_open = open

    def fake_open(path, *args, **kwargs):
        # 让知识库文件读取为空内容，触发 load_knowledge 返回 []
        if "知识库简易版" in str(path):
            return io.StringIO("")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=fake_open):
        # reload 会重新执行模块级初始化代码（含 FAISS 索引创建）
        importlib.reload(agent)

    # 若到达此处说明没有崩溃
    assert agent.knowledge == []
    assert agent.index.ntotal == 0
    # 维度应由占位文本编码得出，而非从空数组取 shape[1]
    assert agent.index.dim == 8

    # 恢复正常状态，避免影响后续测试
    importlib.reload(agent)


# ---------------------------------------------------------------------------
# Bug 2：LLM 返回非法 JSON 工具参数时，/chat 不应崩溃
# ---------------------------------------------------------------------------
def test_rag_chat_handles_invalid_tool_call_arguments_json():
    """模型在 tool_calls.arguments 中返回非法 JSON 时，应返回友好提示而非抛异常。"""
    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": "{query: 重庆天气,}",  # 非法 JSON
                            }
                        }
                    ]
                }
            }
        ]
    }
    with patch("agent.call_llm", return_value=bad_response):
        result = agent.rag_chat("重庆今天天气怎么样？", [])

    assert isinstance(result, str)
    assert "解析失败" in result


def test_rag_chat_handles_null_tool_arguments():
    """arguments 为 None 时不应抛 TypeError。"""
    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "search_web", "arguments": None}}
                    ]
                }
            }
        ]
    }
    with patch("agent.call_llm", return_value=bad_response):
        result = agent.rag_chat("你好", [])

    assert isinstance(result, str)
    assert "解析失败" in result


# ---------------------------------------------------------------------------
# Bug 3：add_knowledge 并发安全（knowledge 列表与 FAISS 索引一致）
# ---------------------------------------------------------------------------
def _reset_knowledge_state():
    agent.knowledge = []
    agent.index = agent.faiss.IndexFlatIP(8)


def test_add_knowledge_is_thread_safe_under_concurrent_uploads():
    """并发调用 add_knowledge 时，knowledge 与 index 数量应一致且不抛异常。"""
    _reset_knowledge_state()

    errors = []
    results = []
    barrier = threading.Barrier(10)

    def worker(i):
        try:
            barrier.wait(timeout=5)
            added, total = agent.add_knowledge([f"chunk_{i}"])
            results.append((added, total))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"并发写入出现异常: {errors}"
    assert len(agent.knowledge) == 10
    assert agent.index.ntotal == 10
    # 顺序一致：knowledge 中第 i 块文本应与 index 中第 i 个向量对应
    # 通过再次搜索验证索引未损坏且可正常工作
    q = agent.model.encode(["probe"])[0]
    dists, idxs = agent.index.search(np.array([q], dtype="float32"), 1)
    assert idxs[0][0] != -1


def test_concurrent_add_and_search_do_not_corrupt_index():
    """并发的 add 与 search 不应导致索引损坏或错位。"""
    _reset_knowledge_state()
    # 先放入一些初始数据
    agent.add_knowledge([f"init_{i}" for i in range(5)])

    errors = []

    def adder():
        try:
            for i in range(5):
                agent.add_knowledge([f"added_{i}"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def searcher():
        try:
            q = agent.model.encode(["query"])[0]
            for _ in range(5):
                agent.search_mostk(q, K=3)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=adder), threading.Thread(target=searcher)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"并发读写出现异常: {errors}"
    assert len(agent.knowledge) == agent.index.ntotal


# ---------------------------------------------------------------------------
# Bug 4：call_llm 必须设置超时，且网络异常应被优雅处理
# ---------------------------------------------------------------------------
def test_call_llm_passes_timeout_to_requests():
    """请求大模型时必须携带 timeout 参数，避免无限挂起。"""
    with patch("agent.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"choices": []}
        mock_post.return_value.status_code = 200
        agent.call_llm("test prompt", "chat")

    _, kwargs = mock_post.call_args
    assert "timeout" in kwargs, "requests.post 未设置 timeout，存在无限挂起风险"
    assert kwargs["timeout"] is not None
    assert kwargs["timeout"] > 0


def test_call_llm_returns_error_dict_on_request_exception():
    """网络异常（连接失败/超时）应返回 error 字典，而非让异常传播导致 500。"""
    with patch(
        "agent.requests.post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        result = agent.call_llm("test prompt", "chat")

    assert isinstance(result, dict)
    assert "error" in result


def test_call_llm_returns_error_dict_on_non_json_response():
    """API 返回非 JSON 响应（如网关错误页）时不应抛异常。"""
    with patch("agent.requests.post") as mock_post:
        mock_post.return_value.status_code = 502
        mock_post.return_value.json.side_effect = ValueError("Expecting value")
        # 传入 tools 强制走非流式分支，才会调用 response.json()
        result = agent.call_llm("test prompt", "chat", tools=[])

    assert isinstance(result, dict)
    assert "error" in result
