"""
针对 agent.py 中已修复缺陷的回归测试。

由于运行环境未安装 faiss / numpy / text2vec / pymupdf 等重依赖，
这里通过 sys.modules 注入轻量假对象，使 agent 模块可被导入，
并聚焦验证关键缺陷修复行为。
"""
import builtins
import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# 假依赖注入（在 import agent 之前完成）
# ---------------------------------------------------------------------------

def _install_fakes():
    # dotenv
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv

    # colorlog（logger_utils 依赖）
    import logging

    class FakeColoredFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, log_colors=None, **kwargs):
            # 去掉 colorlog 特有的 %(log_color)s 占位符，避免格式化报错
            if fmt:
                fmt = fmt.replace("%(log_color)s", "")
            super().__init__(fmt=fmt, datefmt=datefmt)

    colorlog = types.ModuleType("colorlog")
    colorlog.ColoredFormatter = FakeColoredFormatter
    colorlog.StreamHandler = logging.StreamHandler  # 必须是真实 handler，否则 level 为 MagicMock
    sys.modules["colorlog"] = colorlog

    # requests（仅保证可导入，测试不发起网络请求）
    requests = types.ModuleType("requests")
    requests.post = mock.MagicMock()
    sys.modules["requests"] = requests

    # numpy
    class FakeArray:
        def __init__(self, data):
            self._data = data
            if isinstance(data, list):
                if data and isinstance(data[0], list):
                    self.shape = (len(data), len(data[0]))
                else:
                    self.shape = (len(data),)
            else:
                self.shape = ()

        def astype(self, dtype):
            return self

    class FakeNumpy:
        @staticmethod
        def array(data, dtype=None):
            return FakeArray(data)

    sys.modules["numpy"] = FakeNumpy()

    # faiss
    class FakeIndex:
        def __init__(self, dim):
            self.dim = dim
            self.ntotal = 0
            self._fail_next_add = False

        def add(self, vecs):
            if self._fail_next_add:
                self._fail_next_add = False
                raise RuntimeError("simulated faiss add failure")
            n = vecs.shape[0]
            self.ntotal += n

        def search(self, query_vecs, k):
            real = min(k, self.ntotal)
            distances = [[0.9] * real + [-1.0] * (k - real)]
            indices = [list(range(real)) + [-1] * (k - real)]
            return distances, indices

    class FakeFaiss:
        IndexFlatIP = FakeIndex

        @staticmethod
        def normalize_L2(vecs):
            return None

    sys.modules["faiss"] = FakeFaiss()

    # text2vec
    text2vec = types.ModuleType("text2vec")

    class FakeSentenceModel:
        def encode(self, texts):
            # 每个文本返回一个 8 维假向量
            return [[0.1] * 8 for _ in texts]

    text2vec.SentenceModel = FakeSentenceModel
    sys.modules["text2vec"] = text2vec

    # langchain_text_splitters
    lts = types.ModuleType("langchain_text_splitters")

    class FakeSplitter:
        def __init__(self, **kwargs):
            self.chunk_size = kwargs.get("chunk_size", 100)

        def split_text(self, text):
            if not text:
                return []
            # 简单按字符数切分，便于测试
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    lts.RecursiveCharacterTextSplitter = FakeSplitter
    sys.modules["langchain_text_splitters"] = lts

    # tavily
    tavily = types.ModuleType("tavily")
    tavily.TavilyClient = mock.MagicMock
    sys.modules["tavily"] = tavily

    # docx
    docx = types.ModuleType("docx")
    docx.Document = mock.MagicMock
    sys.modules["docx"] = docx

    # fitz
    fitz = types.ModuleType("fitz")

    class FakePage:
        def __init__(self, text, raise_on_get=False):
            self._text = text
            self._raise = raise_on_get

        def get_text(self):
            if self._raise:
                raise RuntimeError("corrupt page text extraction")
            return self._text

    class FakeDoc:
        instances = []

        def __init__(self, pages):
            self._pages = pages
            self.closed = False
            FakeDoc.instances.append(self)

        def __iter__(self):
            return iter(self._pages)

        def close(self):
            self.closed = True

    fitz.FileDataError = type("FileDataError", (Exception,), {})

    def fake_open(stream=None, filetype=None):
        # 测试通过设置 fitz._next_doc 来控制返回的文档
        pages = getattr(fitz, "_next_pages", [FakePage("hello"), FakePage("world")])
        return FakeDoc(pages)

    fitz.open = fake_open
    fitz.FakeDoc = FakeDoc
    fitz.FakePage = FakePage
    sys.modules["fitz"] = fitz


_install_fakes()

import agent  # noqa: E402


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestCalculateDoS(unittest.TestCase):
    """修复：calculate 不得因幂运算/超长表达式导致资源耗尽。"""

    def test_rejects_power_operator(self):
        # 9**9**9 会被计算为超大整数，导致 CPU/内存耗尽
        result = agent.calculate("9**9**9")
        self.assertIn("不支持幂运算", result)

    def test_rejects_long_expression(self):
        result = agent.calculate("1+" * 200)
        self.assertIn("过长", result)

    def test_accepts_normal_expression(self):
        result = agent.calculate("2 + 3 * 4")
        self.assertIn("14", result)

    def test_rejects_disallowed_chars(self):
        result = agent.calculate("__import__('os')")
        self.assertIn("不支持的字符", result)


class TestParsePdfLeak(unittest.TestCase):
    """修复：parse_pdf 在解析异常时仍需关闭 fitz 文档，避免内存泄漏。"""

    def test_closes_doc_when_page_raises(self):
        import fitz

        fitz.FakeDoc.instances.clear()
        fitz._next_pages = [fitz.FakePage("ok"), fitz.FakePage("", raise_on_get=True)]

        with self.assertRaises(RuntimeError):
            agent.parse_pdf(b"fake pdf bytes")

        # 即使抛异常，doc.close() 也必须被调用
        docs = fitz.FakeDoc.instances
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0].closed, "fitz 文档在异常路径下未被关闭，存在内存泄漏")

    def test_closes_doc_on_success(self):
        import fitz

        fitz.FakeDoc.instances.clear()
        fitz._next_pages = [fitz.FakePage("hello"), fitz.FakePage("world")]

        text = agent.parse_pdf(b"fake pdf bytes")
        self.assertEqual(text, "helloworld")
        self.assertTrue(fitz.FakeDoc.instances[0].closed)


class TestKnowledgeConsistency(unittest.TestCase):
    """修复：add_knowledge 中 index.add 与 knowledge.extend 不得错位。"""

    def setUp(self):
        # 保存并重置共享状态
        self._orig_index = agent.index
        self._orig_knowledge = agent.knowledge
        agent.knowledge = []
        agent.index = agent.faiss.IndexFlatIP(8)

    def tearDown(self):
        agent.index = self._orig_index
        agent.knowledge = self._orig_knowledge

    def test_index_add_failure_does_not_desync(self):
        # 若 index.add 抛异常，knowledge 列表不得被修改，
        # 否则后续检索会返回索引错位的知识块（静默数据损坏）。
        agent.index._fail_next_add = True
        before = len(agent.knowledge)

        with self.assertRaises(RuntimeError):
            agent.add_knowledge(["chunk-a", "chunk-b"])

        self.assertEqual(len(agent.knowledge), before, "index.add 失败后 knowledge 不应变化")

    def test_add_creates_index_when_none(self):
        # 知识库为空时 index 为 None，首次 add 应自动创建索引
        agent.index = None
        added, total = agent.add_knowledge(["first chunk"])
        self.assertEqual(added, 1)
        self.assertEqual(total, 1)
        self.assertIsNotNone(agent.index)
        self.assertEqual(agent.index.ntotal, 1)


class TestEmptyKnowledgeStartup(unittest.TestCase):
    """修复：知识库为空/缺失时应用不得在导入阶段崩溃。"""

    def test_search_returns_empty_when_index_none(self):
        orig_index = agent.index
        try:
            agent.index = None
            # 不应抛出异常，直接返回空列表
            self.assertEqual(agent.search_mostk([0.1] * 8), [])
        finally:
            agent.index = orig_index

    def test_module_reload_with_empty_knowledge_does_not_crash(self):
        # 模拟知识库文件为空：重载 agent 模块，验证不再因 shape[1] 抛 IndexError
        real_open = builtins.open

        class EmptyFile:
            def read(self):
                return ""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("知识库简易版.txt"):
                return EmptyFile()
            return real_open(path, *args, **kwargs)

        with mock.patch.object(builtins, "open", side_effect=fake_open):
            importlib.reload(agent)

        # 知识库为空时 index 应为 None，而不是崩溃
        self.assertIsNone(agent.index)
        self.assertEqual(agent.knowledge, [])


class TestConcurrency(unittest.TestCase):
    """修复：并发 add/search 必须通过锁保证 index 与 knowledge 一致。"""

    def setUp(self):
        self._orig_index = agent.index
        self._orig_knowledge = agent.knowledge
        agent.knowledge = []
        agent.index = agent.faiss.IndexFlatIP(8)

    def tearDown(self):
        agent.index = self._orig_index
        agent.knowledge = self._orig_knowledge

    def test_concurrent_adds_keep_index_and_list_in_sync(self):
        errors = []

        def worker(i):
            try:
                agent.add_knowledge([f"chunk-{i}"])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"并发写入出现异常: {errors}")
        # 关键断言：knowledge 列表长度必须与 FAISS 索引向量数完全一致
        self.assertEqual(len(agent.knowledge), agent.index.ntotal)
        self.assertEqual(len(agent.knowledge), 50)

    def test_concurrent_add_and_search_no_crash(self):
        stop = threading.Event()
        errors = []

        def adder():
            try:
                for i in range(30):
                    agent.add_knowledge([f"add-{i}"])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def searcher():
            try:
                while not stop.is_set():
                    agent.search_mostk([0.1] * 8, K=4)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t_add = threading.Thread(target=adder)
        t_search = threading.Thread(target=searcher)
        t_add.start()
        t_search.start()
        t_add.join()
        stop.set()
        t_search.join(timeout=2)

        self.assertEqual(errors, [], f"并发读写出现异常: {errors}")
        self.assertEqual(len(agent.knowledge), agent.index.ntotal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
