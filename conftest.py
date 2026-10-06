"""测试配置：在导入 agent 前桩掉重型第三方依赖。

agent.py 在模块顶层会加载 SentenceModel、创建 FAISS 索引等，这些依赖在
测试环境中通常不可用。通过在 sys.modules 中注入桩模块，可以让 agent.py
顺利导入，从而对其中的纯 Python 函数（如 calculate、search_mostk、
rag_chat 的提示词构造逻辑）进行单元测试。
"""
import os
import sys
import types
from unittest.mock import MagicMock

# 确保测试从项目根目录运行，使 load_knowledge("知识库简易版.txt") 能找到文件
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _make_numpy_mock():
    """最小化的 numpy 桩，支持 array().astype().shape[1]。"""
    np = types.ModuleType("numpy")

    class _Array:
        def __init__(self, data=None):
            self._data = data if data is not None else []

        def astype(self, _dtype):
            return self

        @property
        def shape(self):
            n = len(self._data) if hasattr(self._data, "__len__") else 1
            return (n, 768)

        def __len__(self):
            return len(self._data)

    def _array(x):
        return _Array(x)

    np.array = _array
    np.ndarray = _Array
    return np


def _make_faiss_mock():
    faiss = types.ModuleType("faiss")
    _index = MagicMock()
    _index.ntotal = 0
    faiss.IndexFlatIP = MagicMock(return_value=_index)
    faiss.normalize_L2 = MagicMock()
    return faiss


def _make_text2vec_mock():
    text2vec = types.ModuleType("text2vec")
    _model = MagicMock()
    # encode 返回与输入条数相同数量的 768 维向量列表
    _model.encode = lambda items: [[0.0] * 768 for _ in range(len(items))]
    text2vec.SentenceModel = MagicMock(return_value=_model)
    return text2vec


def _stub(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# --- 在 agent.py 被导入前注入所有桩模块 ---
sys.modules["dotenv"] = _stub("dotenv", {"load_dotenv": lambda: None})
sys.modules["requests"] = _stub("requests")
sys.modules["numpy"] = _make_numpy_mock()
sys.modules["fitz"] = _stub("fitz")
sys.modules["text2vec"] = _make_text2vec_mock()

_lts = _stub("langchain_text_splitters")


class _FakeSplitter:
    def __init__(self, **kwargs):
        pass

    def split_text(self, text):
        return [text] if text else []


_lts.RecursiveCharacterTextSplitter = _FakeSplitter
sys.modules["langchain_text_splitters"] = _lts

sys.modules["tavily"] = _stub("tavily", {"TavilyClient": MagicMock()})
sys.modules["docx"] = _stub("docx", {"Document": MagicMock()})
sys.modules["faiss"] = _make_faiss_mock()
# 桩掉 logger_utils，避免创建真实日志文件
sys.modules["logger_utils"] = _stub("logger_utils", {"logger": MagicMock()})
