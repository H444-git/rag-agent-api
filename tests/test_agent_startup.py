import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# 知识库文件路径与 agent.py 中保持一致
_KNOWLEDGE_FILE = "知识库简易版.txt"


def _mock_heavy_modules():
    """模拟 text2vec 等重依赖，使 agent.py 可在不下载模型的情况下被导入测试。"""
    fake_text2vec = MagicMock()
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1] * 384])
    fake_text2vec.SentenceModel.return_value = fake_model

    return {
        "text2vec": fake_text2vec,
        "langchain_text_splitters": MagicMock(),
        "tavily": MagicMock(),
        "docx": MagicMock(),
        "fitz": MagicMock(),
        "requests": MagicMock(),
    }


def _selective_open(real_open, file_path, content=None):
    """构造一个仅拦截指定知识库文件路径的 open 替代函数。"""

    def wrapper(path, *args, **kwargs):
        if path == file_path:
            if content is None:
                raise FileNotFoundError(path)
            from io import StringIO

            return StringIO(content)
        return real_open(path, *args, **kwargs)

    return wrapper


def test_agent_starts_with_missing_knowledge_file():
    """当知识库文件缺失时，agent.py 仍能完成导入并创建空 FAISS 索引，不会触发 IndexError。"""
    fake_modules = _mock_heavy_modules()
    sys.modules.pop("agent", None)

    with patch.dict("sys.modules", fake_modules):
        with patch(
            "builtins.open",
            _selective_open(open, _KNOWLEDGE_FILE, content=None),
        ):
            import agent

            assert agent.knowledge == []
            assert agent.index is not None
            assert agent.index.ntotal == 0
            assert agent.index.d == 384
