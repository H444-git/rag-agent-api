import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


_KNOWLEDGE_FILE = "知识库简易版.txt"


def _mock_heavy_modules():
    fake_text2vec = MagicMock()
    fake_model = MagicMock()

    def fake_encode(texts):
        # 根据输入长度返回对应数量的 384 维向量
        if isinstance(texts, str):
            return np.array([0.1] * 384)
        n = len(texts) if hasattr(texts, "__len__") else 1
        return np.array([[0.1] * 384] * n)

    fake_model.encode.side_effect = fake_encode
    fake_text2vec.SentenceModel.return_value = fake_model

    return {
        "text2vec": fake_text2vec,
        "langchain_text_splitters": MagicMock(),
        "tavily": MagicMock(),
        "docx": MagicMock(),
        "fitz": MagicMock(),
        "requests": MagicMock(),
    }


def test_concurrent_add_and_search():
    """并发 add_knowledge 与 search_mostk 不应抛出异常或返回越界索引。"""
    fake_modules = _mock_heavy_modules()
    sys.modules.pop("agent", None)

    with patch.dict("sys.modules", fake_modules):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "初始知识。"
            import agent

    errors = []

    def worker_add():
        try:
            for _ in range(20):
                agent.add_knowledge(["新知识点"] * 3)
        except Exception as e:
            errors.append(e)

    def worker_search():
        try:
            for _ in range(20):
                agent.search_mostk(np.array([0.1] * 384))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_add) for _ in range(3)] + [
        threading.Thread(target=worker_search) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发测试出现错误: {errors}"
    assert agent.index.ntotal == len(agent.knowledge)
