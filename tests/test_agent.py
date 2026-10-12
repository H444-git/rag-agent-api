import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 在导入 agent 之前 mock 掉重量依赖，避免加载模型和真实网络客户端
EMBEDDING_DIM = 384

_mock_faiss = MagicMock()
_mock_index = MagicMock()
_mock_index.ntotal = 0
_mock_faiss.IndexFlatIP.return_value = _mock_index
_mock_faiss.normalize_L2 = MagicMock()
sys.modules["faiss"] = _mock_faiss

_mock_text2vec = MagicMock()
_mock_model = MagicMock()


def _mock_encode(inputs):
    if isinstance(inputs, str):
        count = 1
    else:
        count = len(inputs)
    return np.random.rand(count, EMBEDDING_DIM).astype("float32")


_mock_model.encode.side_effect = _mock_encode
_mock_text2vec.SentenceModel.return_value = _mock_model
sys.modules["text2vec"] = _mock_text2vec

_mock_tavily = MagicMock()
_mock_tavily_client = MagicMock()
_mock_tavily.TavilyClient.return_value = _mock_tavily_client
sys.modules["tavily"] = _mock_tavily

sys.modules["fitz"] = MagicMock()
sys.modules["docx"] = MagicMock()

_mock_splitters = MagicMock()
_mock_splitter = MagicMock()
_mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
_mock_splitters.RecursiveCharacterTextSplitter.return_value = _mock_splitter
sys.modules["langchain_text_splitters"] = _mock_splitters

sys.modules["requests"] = MagicMock()
sys.modules["colorlog"] = MagicMock()

with patch.dict(
    "os.environ", {"DEEPSEEK_API_KEY": "test-key", "TAVILY_API_KEY": "test-key"}
):
    import agent


def test_empty_knowledge_initializes_empty_index():
    """当 load_knowledge 返回空列表时，模块初始化不应因 shape[1] 越界而崩溃。"""
    import importlib

    with patch("agent.load_knowledge", return_value=[]):
        importlib.reload(agent)

    assert agent.index is _mock_index
    _mock_faiss.IndexFlatIP.assert_called_with(EMBEDDING_DIM)


def test_add_knowledge_thread_safety():
    """并发 add_knowledge 应保持 knowledge 与 FAISS index 的一致性。"""
    import importlib

    importlib.reload(agent)
    _mock_index.reset_mock()

    def worker(i):
        agent.add_knowledge([f"chunk-{i}-a", f"chunk-{i}-b"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 加锁后 10 个线程各添加 2 块，最终应为 20 块
    assert len(agent.knowledge) == 20
    assert _mock_index.add.call_count == 10


def test_rag_chat_rejects_malformed_tool_json():
    """模型返回非法 JSON 的工具参数时，不应触发未捕获的 JSONDecodeError。"""
    bad_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": "not valid json",
                            }
                        }
                    ]
                }
            }
        ]
    }

    with patch("agent.call_llm", return_value=bad_result):
        answer = agent.rag_chat("test question", [])

    assert "解析失败" in answer


def test_rag_chat_rejects_missing_tool_args():
    """工具参数缺少必需的 query/expression 时，不应触发 KeyError。"""
    bad_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": "{}",
                            }
                        }
                    ]
                }
            }
        ]
    }

    with patch("agent.call_llm", return_value=bad_result):
        answer = agent.rag_chat("test question", [])

    assert "缺少有效" in answer


def test_calculate_blocks_disallowed_chars():
    """calculate 应拒绝包含非预期字符的表达式，避免 eval 被绕过。"""
    result = agent.calculate("__import__('os').system('id')")
    assert "不支持的字符" in result


def test_calculate_executes_safe_expression():
    result = agent.calculate("(1 + 2) * 3")
    assert "9" in result
