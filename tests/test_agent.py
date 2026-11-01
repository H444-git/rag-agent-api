import importlib
import os
import threading
import time
from pathlib import Path

import numpy as np

import agent


def _normalized(vec):
    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def test_calculate_rejects_exponential_blowup():
    # 9**9**9 会让 Python 计算 3 亿多位的超大整数，足以卡死进程
    result = agent.calculate("9**9**9")
    assert result.startswith("错误")
    assert "幂运算指数过大" in result


def test_calculate_keeps_small_power_and_basic_math():
    assert "1024" in agent.calculate("2**10")
    assert "7" in agent.calculate("1+2*3")


def test_add_knowledge_with_empty_chunks_is_a_noop():
    before = len(agent.get_all_knowledge())
    added, total = agent.add_knowledge([])
    assert added == 0
    assert total == before


class _BlockingIndex:
    """在 index.add 里阻塞，用于构造"上传进行中"的时间窗口"""

    def __init__(self, dim):
        self.d = dim
        self.ntotal = 0
        self.added = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def add(self, vecs):
        self.entered.set()
        self.release.wait(timeout=5)
        vecs = np.asarray(vecs, dtype="float32")
        self.added.append(vecs.copy())
        self.ntotal += len(vecs)

    def search(self, x, k):
        raise NotImplementedError


def test_concurrent_adds_keep_knowledge_and_index_in_sync():
    old_knowledge = agent.knowledge
    old_index = agent.index
    dim = len(agent.model.encode(["x"])[0])
    agent.knowledge = []
    agent.index = _BlockingIndex(dim)
    blocking = agent.index
    errors = []

    def run(chunks):
        try:
            agent.add_knowledge(chunks)
        except Exception as exc:  # noqa: BLE001 - 测试需要捕获线程内异常
            errors.append(exc)

    t1 = threading.Thread(target=run, args=(["甲块"],))
    t2 = threading.Thread(target=run, args=(["乙块"],))
    try:
        t1.start()
        assert blocking.entered.wait(timeout=5), "第一个上传没有进入索引写入"
        t2.start()
        time.sleep(0.2)
        # 第一个线程正在 index.add 中；若没有互斥锁，第二个线程会抢先 extend，
        # 之后索引下标就会与 knowledge 错位，检索时返回错误的知识块
        assert len(agent.knowledge) == 1
        blocking.release.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, errors
        assert agent.knowledge == ["甲块", "乙块"]
        assert np.allclose(blocking.added[0][0], _normalized(agent.model.encode(["甲块"])[0]))
        assert np.allclose(blocking.added[1][0], _normalized(agent.model.encode(["乙块"])[0]))
    finally:
        blocking.release.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        agent.knowledge = old_knowledge
        agent.index = old_index


def test_call_llm_handles_network_failure(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        raise agent.requests.exceptions.Timeout("连接超时")

    monkeypatch.setattr(agent.requests, "post", fake_post)
    result = agent.call_llm("你好", "chat", tools=agent.tools)
    assert captured["timeout"] is not None
    assert isinstance(result, dict) and "error" in result


class _ErrorResponse:
    status_code = 500

    def json(self):
        return {"error": "internal"}


def test_call_llm_handles_http_error(monkeypatch):
    monkeypatch.setattr(agent.requests, "post", lambda url, **kwargs: _ErrorResponse())
    result = agent.call_llm("你好", "chat", tools=agent.tools)
    assert isinstance(result, dict) and "error" in result


def test_startup_with_missing_knowledge_file_does_not_crash(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    # 缺少 知识库简易版.txt 时，模块导入不应因为空索引维度而崩溃
    importlib.reload(agent)
    assert agent.knowledge == []
    assert agent.index is None
    os.chdir(repo_root)
    importlib.reload(agent)
    assert agent.knowledge != []
    assert agent.index is not None
