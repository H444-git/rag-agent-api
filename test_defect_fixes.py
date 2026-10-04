"""针对本次提交修复的高影响缺陷的回归测试。

通过桩掉 text2vec / faiss / tavily 等重依赖，直接对纯逻辑函数进行验证：
1. calculate() 不再使用 eval，必须拒绝沙箱逃逸载荷。
2. rag_chat() 解析模型工具调用响应时，面对缺键/空 choices/非法 JSON
   不得抛出未捕获异常（避免 /chat 请求 500）。
3. parse_pdf() 在解析过程抛异常时仍会关闭文档（资源不泄漏）。
"""
import sys
import types
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# 在导入 agent 之前，桩掉所有重量级第三方依赖，使测试可在无 GPU/无模型/
# 无 API Key 的环境下运行。
# ---------------------------------------------------------------------------
def _install_stubs():
    stubs = {
        "dotenv": types.ModuleType("dotenv"),
        "requests": types.ModuleType("requests"),
        "numpy": types.ModuleType("numpy"),
        "fitz": types.ModuleType("fitz"),
        "faiss": types.ModuleType("faiss"),
        "text2vec": types.ModuleType("text2vec"),
        "langchain_text_splitters": types.ModuleType("langchain_text_splitters"),
        "tavily": types.ModuleType("tavily"),
        "docx": types.ModuleType("docx"),
        "colorlog": types.ModuleType("colorlog"),
    }

    stubs["dotenv"].load_dotenv = lambda *a, **k: None

    class _FakeArray:
        def __init__(self, data):
            self._data = data
            # data 期望是二维列表 [[...], ...]
            try:
                self.shape = (len(data), len(data[0]) if data else 0)
            except TypeError:
                self.shape = (1, len(data))

        def astype(self, _):
            return self

        def __len__(self):
            return len(self._data)

    stubs["numpy"].array = lambda x, *a, **k: _FakeArray(x)
    stubs["numpy"].float32 = "float32"

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def json(self):
            return self._payload
        def iter_lines(self):
            return iter([])

    stubs["requests"].post = lambda *a, **k: _FakeResp({})

    class _Doc:
        def __init__(self, pages):
            self._pages = pages
            self.closed = False
        def __iter__(self):
            return iter(self._pages)
        def close(self):
            self.closed = True

    class _Page:
        def __init__(self, text, fail=False):
            self._text = text
            self._fail = fail
        def get_text(self):
            if self._fail:
                raise RuntimeError("boom")
            return self._text

    stubs["fitz"]._Doc = _Doc
    stubs["fitz"]._Page = _Page
    stubs["fitz"].open = lambda *a, **k: _Doc([_Page("hello")])

    class _Idx:
        def __init__(self):
            self.ntotal = 0
        def add(self, v):
            self.ntotal += len(v)
        def search(self, v, k):
            return [[0.9]], [[0]]
    stubs["faiss"].IndexFlatIP = lambda d: _Idx()
    stubs["faiss"].normalize_L2 = lambda v: None

    class _Model:
        def encode(self, texts):
            if isinstance(texts, str):
                return [0.1]
            return [[0.1] for _ in texts]
    stubs["text2vec"].SentenceModel = lambda *a, **k: _Model()

    class _Splitter:
        def __init__(self, *a, **k):
            pass
        def split_text(self, text):
            return [text] if text else []
    stubs["langchain_text_splitters"].RecursiveCharacterTextSplitter = _Splitter

    class _Tavily:
        def __init__(self, *a, **k):
            pass
        def search(self, *a, **k):
            return {"results": [], "answer": ""}
    stubs["tavily"].TavilyClient = _Tavily

    class _Docx:
        def __init__(self, *a, **k):
            self.paragraphs = []
    stubs["docx"].Document = _Docx

    stubs["colorlog"].ColoredFormatter = lambda *a, **k: None
    import logging as _logging
    stubs["colorlog"].StreamHandler = lambda *a, **k: _logging.StreamHandler()

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


_install_stubs()

import agent  # noqa: E402


class CalculateSecurityTests(unittest.TestCase):
    """缺陷 1：calculate 不得执行任意代码。"""

    def test_normal_arithmetic(self):
        self.assertEqual(agent.calculate("2 + 3 * 4"), "计算结果：2 + 3 * 4 = 14")

    def test_division_by_zero_is_reported(self):
        out = agent.calculate("1 / 0")
        self.assertIn("除数不能为零", out)

    def test_sandbox_escape_payload_is_rejected(self):
        # 经典 eval 沙箱逃逸链：通过 object.__subclasses__ 拿到 os 模块。
        payload = (
            "().__class__.__bases__[0].__subclasses__()"
        )
        out = agent.calculate(payload)
        self.assertTrue(
            out.startswith("错误") or out.startswith("计算错误"),
            f"逃逸载荷未被拒绝: {out}",
        )

    def test_arbitrary_attribute_access_blocked(self):
        out = agent.calculate("(1).__class__")
        self.assertTrue(out.startswith("错误") or out.startswith("计算错误"))

    def test_non_string_input(self):
        self.assertIn("错误", agent.calculate(None))
        self.assertIn("错误", agent.calculate(""))


class RagChatRobustnessTests(unittest.TestCase):
    """缺陷 2：rag_chat 解析模型响应不得因缺键/非法 JSON 崩溃。"""

    def setUp(self):
        # 保证知识库非空，使函数进入检索分支。
        agent.knowledge = ["块1 测试知识"]
        agent.index = mock.MagicMock()
        agent.index.search.return_value = ([[0.9]], [[0]])

    def _run(self, llm_response):
        with mock.patch.object(agent, "call_llm", return_value=llm_response):
            # 不应抛出任何异常
            return agent.rag_chat("你好", [])

    def test_empty_choices_does_not_crash(self):
        result = self._run({"choices": []})
        self.assertIsInstance(result, str)

    def test_missing_choices_does_not_crash(self):
        result = self._run({})
        self.assertIsInstance(result, str)

    def test_tool_call_with_invalid_json_does_not_crash(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "calculate",
                                    "arguments": "{not valid json",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        result = self._run(resp)
        self.assertIsInstance(result, str)
        self.assertIn("参数格式错误", result)

    def test_tool_call_missing_function_name(self):
        resp = {
            "choices": [
                {"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}
            ]
        }
        result = self._run(resp)
        self.assertIsInstance(result, str)

    def test_direct_answer_path(self):
        resp = {"choices": [{"message": {"content": "直接回答"}}]}
        result = self._run(resp)
        self.assertEqual(result, "直接回答")

    def test_error_response(self):
        result = self._run({"error": "模型过载"})
        self.assertEqual(result, "模型过载")


class ParsePdfResourceTests(unittest.TestCase):
    """缺陷 3：parse_pdf 异常时必须关闭文档。"""

    def test_document_closed_on_success(self):
        page = agent.fitz._Page("hello")
        doc = agent.fitz._Doc([page])
        with mock.patch.object(agent.fitz, "open", return_value=doc):
            text = agent.parse_pdf(b"fake")
        self.assertEqual(text, "hello")
        self.assertTrue(doc.closed)

    def test_document_closed_when_page_raises(self):
        page = agent.fitz._Page("", fail=True)
        doc = agent.fitz._Doc([page])
        with mock.patch.object(agent.fitz, "open", return_value=doc):
            with self.assertRaises(RuntimeError):
                agent.parse_pdf(b"fake")
        self.assertTrue(doc.closed, "文档在解析异常时未被关闭，存在句柄泄漏")


if __name__ == "__main__":
    unittest.main(verbosity=2)
