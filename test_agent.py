"""提交后正确性检查：覆盖已修复的高影响缺陷。

通过桩模块隔离 text2vec/faiss 等重型依赖，专注验证缺陷行为本身。
"""
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


HEAVY_MODULES = [
    "dotenv",
    "requests",
    "colorlog",
    "text2vec",
    "fitz",
    "tavily",
    "docx",
    "langchain_text_splitters",
    "faiss",
]


def _install_stubs():
    """为重型第三方依赖安装桩模块，避免在测试环境下载模型。"""
    for name in HEAVY_MODULES:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    faiss = MagicMock()
    faiss.normalize_L2 = MagicMock()

    def _index_factory(dim):
        idx = MagicMock()
        idx.ntotal = 0
        return idx

    faiss.IndexFlatIP = MagicMock(side_effect=_index_factory)
    sys.modules["faiss"] = faiss

    # text2vec.SentenceModel -> mock model，encode 返回可控向量
    model = MagicMock()

    def _encode(texts, *args, **kwargs):
        n = len(texts)
        return np.zeros((n, 8), dtype="float32")

    model.encode = MagicMock(side_effect=_encode)
    sys.modules["text2vec"].SentenceModel = MagicMock(return_value=model)

    # langchain 分块器：返回真实列表，空文本返回空列表
    splitter = MagicMock()
    splitter.split_text = MagicMock(side_effect=lambda text: [text] if text else [])
    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock(
        return_value=splitter
    )

    # colorlog：StreamHandler 需要数值 level，否则 logging 分发会 TypeError
    import logging

    stream_handler = MagicMock()
    stream_handler.level = logging.NOTSET
    sys.modules["colorlog"].StreamHandler = MagicMock(return_value=stream_handler)
    sys.modules["colorlog"].ColoredFormatter = MagicMock()

    sys.modules["dotenv"].load_dotenv = MagicMock()


def _reload_agent_with_file_content(content):
    """重新加载 agent 模块，并控制知识库文件的读取内容。"""
    _install_stubs()
    if "agent" in sys.modules:
        del sys.modules["agent"]
    from io import StringIO

    fake_file = StringIO(content)
    with patch("builtins.open", return_value=fake_file):
        agent = importlib.import_module("agent")
    return agent


# ---------------------------------------------------------------------------
# 缺陷 1：知识库为空时启动崩溃（np.array([]).shape[1] -> IndexError）
# ---------------------------------------------------------------------------
def test_empty_knowledge_does_not_crash_at_startup():
    """知识库文件缺失或为空时，模块导入不应抛出 IndexError。"""
    agent = _reload_agent_with_file_content("")

    # 索引必须被创建，且维度来自占位文本编码，而不是崩溃
    assert agent.index is not None
    sys.modules["faiss"].IndexFlatIP.assert_called_once()
    called_dim = sys.modules["faiss"].IndexFlatIP.call_args.args[0]
    assert called_dim == 8  # 与桩 model.encode 返回的维度一致
    assert agent.index.ntotal == 0


def test_nonempty_knowledge_initializes_index():
    """回归：知识库非空时索引仍按原逻辑正常构建。"""
    agent = _reload_agent_with_file_content("RAG 是一种检索增强生成技术。")
    assert agent.index is not None
    sys.modules["faiss"].IndexFlatIP.assert_called_once()
    # 非空路径下 normalize_L2 应被调用
    sys.modules["faiss"].normalize_L2.assert_called_once()


# ---------------------------------------------------------------------------
# 缺陷 2：call_llm 请求未设置 timeout，上游无响应时无限挂起
# ---------------------------------------------------------------------------
def test_call_llm_sets_request_timeout():
    """调用大模型时必须传入 timeout，避免连接长时间挂起耗尽工作线程。"""
    agent = _reload_agent_with_file_content("RAG 是一种检索增强生成技术。")

    fake_response = MagicMock()
    fake_response.iter_lines.return_value = iter([])
    fake_response.json.return_value = {}

    with patch.object(agent.requests, "post", return_value=fake_response) as post:
        agent.call_llm("你好", "chat")

    _, kwargs = post.call_args
    assert kwargs.get("timeout") == 60, "requests.post 必须设置 timeout"


# ---------------------------------------------------------------------------
# 缺陷 3：工具调用参数 JSON 解析失败时抛出未处理异常 -> 500
# ---------------------------------------------------------------------------
def test_rag_chat_handles_malformed_tool_arguments():
    """模型返回非法 JSON 的工具参数时，应返回友好提示而非抛出异常。"""
    agent = _reload_agent_with_file_content("RAG 是一种检索增强生成技术。")

    # 构造一个包含非法 JSON 参数的工具调用响应
    malformed_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": "{query: 北京天气",  # 非法 JSON
                            }
                        }
                    ],
                }
            }
        ]
    }

    with patch.object(agent, "search_mostk", return_value=[]), patch.object(
        agent, "call_llm", return_value=malformed_response
    ):
        result = agent.rag_chat("北京今天天气怎么样", [])

    assert isinstance(result, str)
    assert "参数解析失败" in result
