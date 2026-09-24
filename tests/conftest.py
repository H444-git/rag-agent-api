"""测试夹具：在导入 agent 之前注入重依赖的轻量替身。

agent.py 在模块导入时即加载 SentenceModel、构建 FAISS 索引，真实依赖
（faiss / text2vec / pymupdf / docx 等）体积大且沙箱内未安装。这里用
sys.modules 注入最小化 fake，使 agent.py 的逻辑分支可以被单元测试覆盖。
"""
import sys
import types
import logging
from unittest.mock import MagicMock

_VEC_DIM = 768  # 占位向量维度，与真实 text2vec 无关，仅用于让 shape[1] 可取


class FakeArray:
    """最小化 numpy 数组替身，覆盖 agent.py 实际用到的接口。"""

    def __init__(self, shape, ndim, data=None):
        self._shape = tuple(shape)
        self.ndim = ndim
        self._data = data

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        s = 1
        for d in self._shape:
            s *= d
        return s

    def astype(self, _dtype):
        return self

    def __getitem__(self, i):
        if isinstance(self._data, list):
            return self._data[i]
        return self

    def __len__(self):
        return self._shape[0] if self._shape else 0


def _np_array(x, *args, **kwargs):
    """模拟 np.array：根据输入推断 shape/ndim，保持 numpy 数组幂等。"""
    if x is None:
        return FakeArray((0,), 1)
    if isinstance(x, FakeArray):
        return x
    if isinstance(x, list):
        if len(x) == 0:
            return FakeArray((0,), 1)
        first = x[0]
        if isinstance(first, FakeArray):
            d = first._shape[0] if first._shape else 0
            return FakeArray((len(x), d), 2, data=x)
        return FakeArray((len(x),), 1, data=x)
    return FakeArray((0,), 1)


# ---- numpy ----
fake_np = types.SimpleNamespace()
fake_np.array = _np_array
fake_np.ndarray = FakeArray
sys.modules["numpy"] = fake_np


# ---- faiss ----
class FakeIndex:
    """模拟 faiss.IndexFlatIP：记录 add 的条数，search 返回固定命中。"""

    def __init__(self, dim):
        self.dim = dim
        self.ntotal = 0
        self.added = []

    def add(self, arr):
        n = arr.shape[0] if getattr(arr, "shape", ()) else 0
        self.ntotal += max(n, 0)
        self.added.append(arr)

    def search(self, query, k):
        class _Row:
            def __init__(self, vals):
                self._vals = vals

            def __getitem__(self, i):
                return self._vals

        # 固定返回 1 条命中：idx=0，相似度 0.9（>= 默认阈值 0.4）
        return _Row([0.9]), _Row([0])


fake_faiss = types.SimpleNamespace()
fake_faiss.IndexFlatIP = FakeIndex
fake_faiss.normalize_L2 = lambda arr: None
sys.modules["faiss"] = fake_faiss


# ---- text2vec ----
class FakeModel:
    """模拟 SentenceModel：string -> 1 维，list -> 2 维。"""

    def encode(self, sentences):
        if isinstance(sentences, str):
            return FakeArray((_VEC_DIM,), 1)
        if isinstance(sentences, list):
            if len(sentences) == 0:
                return []
            return FakeArray((len(sentences), _VEC_DIM), 2)
        return FakeArray((_VEC_DIM,), 1)


fake_text2vec = types.SimpleNamespace()
fake_text2vec.SentenceModel = lambda: FakeModel()
sys.modules["text2vec"] = fake_text2vec


# ---- langchain_text_splitters ----
class FakeSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_text(self, text):
        if not text:
            return []
        return [text]


fake_lcts = types.SimpleNamespace()
fake_lcts.RecursiveCharacterTextSplitter = FakeSplitter
sys.modules["langchain_text_splitters"] = fake_lcts


# ---- tavily ----
class FakeTavilyClient:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, *args, **kwargs):
        return {"results": [], "answer": ""}


fake_tavily = types.SimpleNamespace()
fake_tavily.TavilyClient = FakeTavilyClient
sys.modules["tavily"] = fake_tavily

# ---- pymupdf (fitz) / python-docx ----
sys.modules["fitz"] = MagicMock()
fake_docx = types.SimpleNamespace()
fake_docx.Document = lambda *a, **k: MagicMock()
sys.modules["docx"] = fake_docx

# ---- requests ----
sys.modules["requests"] = MagicMock()

# ---- dotenv ----
fake_dotenv = types.SimpleNamespace()
fake_dotenv.load_dotenv = lambda: False
sys.modules["dotenv"] = fake_dotenv
sys.modules["dotenv.main"] = fake_dotenv

# ---- colorlog（logger_utils 依赖），用 logging 基类保证 .level 等属性可用 ----
class _Fmt(logging.Formatter):
    def __init__(self, *a, **k):
        super().__init__(fmt="%(message)s")


class _Handler(logging.Handler):
    def __init__(self, *a, **k):
        super().__init__()

    def emit(self, record):
        pass  # 测试中不实际写控制台


fake_colorlog = types.SimpleNamespace()
fake_colorlog.ColoredFormatter = _Fmt
fake_colorlog.StreamHandler = _Handler
sys.modules["colorlog"] = fake_colorlog
