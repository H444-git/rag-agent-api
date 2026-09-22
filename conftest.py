"""
测试配置：在导入 agent 之前，用假的 text2vec 模型替换真实的 SentenceModel，
避免下载/加载大模型。同时确保 tavily 客户端和 DeepSeek 调用都被 mock。
"""
import sys
import types

import numpy as np


class _FakeSentenceModel:
    """模拟 text2vec.SentenceModel：encode 返回固定维度的确定性向量。"""

    DIM = 8

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        # 对每个文本生成一个基于哈希的确定性向量，保证相同文本向量相同
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**32))
            vecs.append(rng.rand(self.DIM).astype("float32"))
        result = np.array(vecs) if vecs else np.array([]).reshape(0, self.DIM)
        # 与 text2vec 行为一致：单个字符串输入返回一维向量
        return result[0] if single else result


# 在 agent 导入 text2vec 之前注入假模块
_text2vec = types.ModuleType("text2vec")
_text2vec.SentenceModel = _FakeSentenceModel
sys.modules["text2vec"] = _text2vec
