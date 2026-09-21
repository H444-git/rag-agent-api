"""测试配置：在导入 agent 前 mock 掉需要下载模型的 text2vec。"""
import sys
from unittest.mock import MagicMock

# mock text2vec.SentenceModel，避免下载模型
text2vec_mock = MagicMock()
sentence_model_mock = MagicMock()

def _fake_encode(texts, **kw):
    np = __import__("numpy")
    if isinstance(texts, str):
        # 单条文本返回 1D 向量 (768,)，与 text2vec 行为一致
        return np.random.rand(768).astype("float32")
    # 文本列表返回 2D 向量 (n, 768)
    return np.random.rand(len(texts), 768).astype("float32")

sentence_model_mock.encode = _fake_encode
text2vec_mock.SentenceModel.return_value = sentence_model_mock
sys.modules.setdefault("text2vec", text2vec_mock)
