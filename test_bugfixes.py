"""
修复回归测试：验证提交中发现的 5 个缺陷已被修复。
仅依赖标准库 + unittest.mock，避免触发 faiss/text2vec 等重依赖。
"""
import sys
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

# 在 mock 后再 import agent，避免触发真实的 faiss/text2vec 加载
for mod in [
    "faiss", "numpy", "text2vec", "fitz", "docx", "requests",
    "tavily", "langchain_text_splitters", "dotenv", "colorlog",
]:
    sys.modules[mod] = mock.MagicMock()
# logger_utils 本身也依赖 colorlog，一并 mock
fake_logger = mock.MagicMock()
sys.modules["logger_utils"] = mock.MagicMock(logger=fake_logger)


def _make_faiss_mock():
    """创建可追踪 add/ntotal 的 FAISS mock index"""
    fake_index = mock.MagicMock()
    fake_index.ntotal = 0
    added_count = [0]

    def fake_add(vecs):
        fake_index.ntotal += vecs.shape[0] if hasattr(vecs, "shape") else len(vecs)
        added_count[0] = fake_index.ntotal

    fake_index.add.side_effect = fake_add
    fake_index.search.return_value = (
        mock.MagicMock(),
        mock.MagicMock(),
    )
    return fake_index, added_count


class EmptyKnowledgeStartupTest(unittest.TestCase):
    """Bug #1: 空知识库 / 文件缺失 不应导致启动崩溃 (IndexError: shape[1])"""

    def test_empty_knowledge_no_crash(self):
        """模拟知识库文件不存在 -> 模块加载不崩"""
        with mock.patch("agent.load_knowledge", return_value=[]):
            import importlib
            import agent as agent_module
            importlib.reload(agent_module)
            # 关键断言：index 应为 None 而非崩溃
            self.assertIsNone(agent_module.index)
            self.assertEqual(agent_module.knowledge, [])


class AddKnowledgeFirstCallTest(unittest.TestCase):
    """Bug #1 附加修复: add_knowledge 在 index=None 时应自行初始化"""

    def test_first_add_when_index_is_none(self):
        import importlib
        import agent as agent_module

        fake_index, _ = _make_faiss_mock()
        with mock.patch.object(agent_module, "index", None), \
             mock.patch.object(agent_module, "_FAISS_DIM", None), \
             mock.patch.object(agent_module, "knowledge", []), \
             mock.patch.object(agent_module, "model") as mock_model, \
             mock.patch("faiss.IndexFlatIP", return_value=fake_index):
            mock_model.encode.return_value = mock.MagicMock()
            mock_model.encode.return_value.shape = (1, 128)  # 1 block, dim 128

            added, total = agent_module.add_knowledge(["hello world"])
            self.assertEqual(added, 1)
            self.assertEqual(total, 1)
            # index 应已自动创建
            self.assertIsNotNone(agent_module.index)


class SearchMostkEmptyIndexTest(unittest.TestCase):
    """search_mostk 在 index=None 或空 knowledge 时应安全返回 []"""

    def test_search_when_index_none(self):
        import agent as agent_module
        with mock.patch.object(agent_module, "index", None), \
             mock.patch.object(agent_module, "knowledge", []):
            result = agent_module.search_mostk([0.1, 0.2, 0.3])
            self.assertEqual(result, [])

    def test_search_when_index_empty(self):
        import agent as agent_module
        fake_index, _ = _make_faiss_mock()
        fake_index.ntotal = 0
        with mock.patch.object(agent_module, "index", fake_index), \
             mock.patch.object(agent_module, "knowledge", []):
            result = agent_module.search_mostk([0.1, 0.2, 0.3])
            self.assertEqual(result, [])


class AddKnowledgeConcurrencyTest(unittest.TestCase):
    """Bug #3: 并发 add_knowledge 后 knowledge 列表长度与 index.ntotal 必须一致"""

    def test_concurrent_adds_preserve_consistency(self):
        import agent as agent_module

        real_knowledge = []
        fake_index, _ = _make_faiss_mock()

        # 让 np.array 返回带真实 shape 的假数组，避免 shape[0] 变成 MagicMock
        class FakeArray:
            def __init__(self, data):
                self._len = len(data) if hasattr(data, "__len__") else 1

            def astype(self, _dt):
                return self

            def normalize_L2(self):
                pass

            @property
            def shape(self):
                return (self._len, 2)

        def fake_array(_data, dtype=None):
            return FakeArray(_data)

        with mock.patch.object(agent_module, "knowledge", real_knowledge), \
             mock.patch.object(agent_module, "index", fake_index), \
             mock.patch.object(agent_module, "_FAISS_DIM", 128), \
             mock.patch.object(agent_module, "np") as mock_np, \
             mock.patch.object(agent_module, "model") as mock_model:
            mock_np.array.side_effect = fake_array
            mock_model.encode.return_value = FakeArray(["x"])

            def add_one(i):
                agent_module.add_knowledge([f"chunk-{i}"])

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(add_one, range(50)))

            self.assertEqual(len(agent_module.knowledge), 50)
            self.assertEqual(agent_module.index.ntotal, 50)


class CallLLMTimeoutTest(unittest.TestCase):
    """Bug #2: call_llm 必须带 timeout 参数"""

    def test_requests_post_has_timeout(self):
        import agent as agent_module

        # 让 call_llm 走完非流式路径（工具调用场景）以验证 timeout
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        mock_response.iter_lines.return_value = []

        with mock.patch.object(agent_module.requests, "post", return_value=mock_response) as mock_post:
            agent_module.call_llm("hi", "chat", tools=[{"type": "function"}])
            kwargs = mock_post.call_args.kwargs
            self.assertIn("timeout", kwargs, "requests.post 缺少 timeout 参数")
            self.assertEqual(kwargs["timeout"], (10, 300))


class CallLLMHTTPErrorTest(unittest.TestCase):
    """Bug #4: call_llm 对 >=400 的 HTTP 状态码应返回 error dict 而非静默/崩溃"""

    def test_401_returns_error_dict(self):
        import agent as agent_module

        mock_response = mock.MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}

        with mock.patch("agent.requests.post", return_value=mock_response):
            result = agent_module.call_llm("hi", "chat", tools=None)
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)
            self.assertIn("401", result["error"])

    def test_500_non_json_body_handled(self):
        import agent as agent_module

        mock_response = mock.MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = ValueError("not json")

        with mock.patch("agent.requests.post", return_value=mock_response):
            result = agent_module.call_llm("hi", "chat", tools=None)
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)

    def test_requests_exception_returns_error_dict(self):
        import agent as agent_module

        class FakeRequestError(Exception):
            pass

        class _Exceptions:
            RequestException = FakeRequestError

        class FakeRequests:
            exceptions = _Exceptions()
            @staticmethod
            def post(*a, **kw):
                raise FakeRequestError("boom")

        with mock.patch.object(agent_module, "requests", FakeRequests):
            result = agent_module.call_llm("hi", "chat", tools=None)
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)
            self.assertIn("网络请求失败", result["error"])


class MainFileStringFormatTest(unittest.TestCase):
    """Bug #5: main.py:46 的 f-string 必须有 f 前缀"""

    def test_fstring_prefix_present(self):
        with open("main.py") as f:
            content = f.read()
        self.assertIn('f"文件上传失败:{e}"', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
