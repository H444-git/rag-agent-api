"""针对历史上遗漏的关键缺陷的回归测试。

agent.py 依赖 text2vec / faiss / tavily 等重型依赖，这里用最小替身替换，
只保留被测逻辑（空知识库初始化、知识库并发读写、表达式校验）。
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeSentenceModel:
    """确定性的向量模型替身"""

    dim = 8

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for col, char in enumerate(text):
                vecs[row, col % self.dim] += (ord(char) % 89) + 1
        return vecs[0] if single else vecs


class FakeSplitter:
    def __init__(self, chunk_size=100, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size

    def split_text(self, text):
        if not text.strip():
            return []
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


class FakeIndexFlatIP:
    """模拟 FAISS IndexFlatIP 的检索语义（内积 + -1 填充）"""

    def __init__(self, dim):
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype="float32")
        self.ntotal = 0

    def add(self, vecs):
        self._vecs = np.vstack([self._vecs, np.asarray(vecs, dtype="float32")])
        self.ntotal = self._vecs.shape[0]

    def search(self, queries, k):
        if self.ntotal == 0:
            return (
                np.zeros((len(queries), k), dtype="float32"),
                np.full((len(queries), k), -1, dtype="int64"),
            )
        sims = np.asarray(queries, dtype="float32") @ self._vecs.T
        order = np.argsort(-sims, axis=1)
        return np.take_along_axis(sims, order, axis=1), order.astype("int64")


def fake_normalize_l2(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms


class FakeTavilyClient:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def search(self, **kwargs):
        return {"results": [], "answer": ""}


def _install_stubs():
    if "colorlog" not in sys.modules:
        colorlog = types.ModuleType("colorlog")

        class ColoredFormatter(logging.Formatter):
            def __init__(self, fmt=None, datefmt=None, log_colors=None, **kwargs):
                super().__init__(fmt=fmt, datefmt=datefmt)

        colorlog.ColoredFormatter = ColoredFormatter
        colorlog.StreamHandler = logging.StreamHandler
        sys.modules["colorlog"] = colorlog

    if "text2vec" not in sys.modules:
        text2vec = types.ModuleType("text2vec")
        text2vec.SentenceModel = FakeSentenceModel
        sys.modules["text2vec"] = text2vec

    if "langchain_text_splitters" not in sys.modules:
        splitters = types.ModuleType("langchain_text_splitters")
        splitters.RecursiveCharacterTextSplitter = FakeSplitter
        sys.modules["langchain_text_splitters"] = splitters

    if "tavily" not in sys.modules:
        tavily = types.ModuleType("tavily")
        tavily.TavilyClient = FakeTavilyClient
        sys.modules["tavily"] = tavily

    if "fitz" not in sys.modules:
        fitz = types.ModuleType("fitz")
        fitz.open = lambda *args, **kwargs: []
        sys.modules["fitz"] = fitz

    if "faiss" not in sys.modules:
        faiss = types.ModuleType("faiss")
        faiss.IndexFlatIP = FakeIndexFlatIP
        faiss.normalize_L2 = fake_normalize_l2
        sys.modules["faiss"] = faiss


def load_agent(monkeypatch, workdir):
    """在指定工作目录下加载 agent 模块（知识库使用相对路径加载）"""
    _install_stubs()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    monkeypatch.chdir(workdir)
    spec = importlib.util.spec_from_file_location("agent_under_test", ROOT / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_startup_with_empty_knowledge_base(monkeypatch, tmp_path):
    """知识库文件缺失时服务仍能启动，并能正常加载与检索新知识"""
    agent = load_agent(monkeypatch, tmp_path)

    assert agent.index.ntotal == 0
    assert agent.get_all_knowledge() == []

    added, total = agent.add_knowledge(["RAG 是检索增强生成"])
    assert (added, total) == (1, 1)

    matched = agent.search_mostk(agent.model.encode("RAG 是检索增强生成"), K=3, threshold=0.0)
    assert matched == ["RAG 是检索增强生成"]


def test_startup_with_existing_knowledge_base(monkeypatch, tmp_path):
    """知识库非空时仍按向量形状推断维度，索引与文本块数量一致"""
    (tmp_path / "知识库简易版.txt").write_text(
        "RAG 是检索增强生成。FAISS 用于向量检索。", encoding="utf-8"
    )

    agent = load_agent(monkeypatch, tmp_path)

    assert agent.index.ntotal == len(agent.get_all_knowledge()) > 0


def test_add_knowledge_ignores_empty_chunks(monkeypatch, tmp_path):
    """空文件分块结果为空时应直接返回，而不是按一维空数组去归一化"""
    agent = load_agent(monkeypatch, tmp_path)

    assert agent.add_knowledge([]) == (0, 0)


def test_get_all_knowledge_returns_snapshot(monkeypatch, tmp_path):
    """返回快照，避免并发写入时序列化抛出 list changed size during iteration"""
    agent = load_agent(monkeypatch, tmp_path)
    agent.add_knowledge(["第一块"])

    snapshot = agent.get_all_knowledge()
    agent.add_knowledge(["第二块"])

    assert snapshot == ["第一块"]


def test_calculate_keeps_normal_expressions(monkeypatch, tmp_path):
    agent = load_agent(monkeypatch, tmp_path)

    assert agent.calculate("(100 + 50) * 0.15").endswith("= 22.5")
    assert agent.calculate("2**10").endswith("= 1024")
    assert agent.calculate("10 % 3").endswith("= 1")


def test_calculate_rejects_unbounded_exponent(monkeypatch, tmp_path):
    """9**9**9 这类表达式会分配数百 MB 整数并长时间占用 CPU，必须拦截"""
    agent = load_agent(monkeypatch, tmp_path)

    assert "计算错误" in agent.calculate("9**9**9")
    assert "计算错误" in agent.calculate("9**387420489")
    assert agent.calculate("错误代码 __import__('os')") == "错误：表达式包含不支持的字符"
