"""为测试注入轻量第三方库桩，避免下载真实 Embedding 模型或安装重型依赖。"""

import logging
import sys
import types

import numpy as np


class _FakeSentenceModel:
    """确定性的假 Embedding 模型：同样文本总是得到同样的向量"""

    dim = 16

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            vecs[i, 0] = 1.0
            for j, ch in enumerate(text):
                vecs[i, (j + 1) % self.dim] += float(ord(ch))
        return vecs


class _FakeIndexFlatIP:
    """行为与 FAISS IndexFlatIP 足够接近的最小实现"""

    def __init__(self, d):
        self.d = d
        self.vecs = np.zeros((0, d), dtype="float32")

    @property
    def ntotal(self):
        return len(self.vecs)

    def add(self, x):
        x = np.atleast_2d(np.asarray(x, dtype="float32"))
        assert x.shape[1] == self.d
        self.vecs = np.vstack([self.vecs, x])

    def search(self, x, k):
        x = np.atleast_2d(np.asarray(x, dtype="float32"))
        n = self.ntotal
        if n == 0:
            return np.zeros((1, k), dtype="float32"), -np.ones((1, k), dtype="int64")
        sims = x @ self.vecs.T
        order = np.argsort(-sims, axis=1)[:, :k]
        dist = np.take_along_axis(sims, order, axis=1).astype("float32")
        if k > n:
            pad = k - n
            dist = np.pad(dist, ((0, 0), (0, pad)))
            order = np.pad(order, ((0, 0), (0, pad)), constant_values=-1)
        return dist, order.astype("int64")


def _normalize_l2(x):
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x /= norms


def _install_stubs():
    text2vec = types.ModuleType("text2vec")
    text2vec.SentenceModel = _FakeSentenceModel

    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = _FakeIndexFlatIP
    faiss.normalize_L2 = _normalize_l2

    class _FakeDoc:
        def __init__(self, *args, **kwargs):
            self._pages = []

        def __iter__(self):
            return iter(self._pages)

        def close(self):
            pass

    fitz = types.ModuleType("fitz")
    fitz.open = lambda *args, **kwargs: _FakeDoc()

    class _FakeSplitter:
        def __init__(self, chunk_size=100, chunk_overlap=0, separators=None):
            self.chunk_size = chunk_size

        def split_text(self, text):
            if not text:
                return []
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

    splitters = types.ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = _FakeSplitter

    class _FakeTavilyClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def search(self, **kwargs):
            return {"results": [], "answer": ""}

    tavily = types.ModuleType("tavily")
    tavily.TavilyClient = _FakeTavilyClient

    class _FakeColoredFormatter(logging.Formatter):
        def __init__(self, *args, **kwargs):
            kwargs.pop("log_colors", None)
            super().__init__(*args, **kwargs)

    colorlog = types.ModuleType("colorlog")
    colorlog.ColoredFormatter = _FakeColoredFormatter
    colorlog.StreamHandler = lambda *args, **kwargs: logging.StreamHandler()

    sys.modules.update(
        {
            "text2vec": text2vec,
            "faiss": faiss,
            "fitz": fitz,
            "langchain_text_splitters": splitters,
            "tavily": tavily,
            "colorlog": colorlog,
        }
    )


_install_stubs()
