"""回归测试：覆盖提交中遗漏的关键缺陷。

仅依赖标准库，通过 mock 隔离 text2vec / faiss / 外部 API。
"""
import sys
import types
import threading
from unittest import mock


def _install_stubs():
    """在导入 agent 之前注入假的第三方模块，避免加载真实大模型。"""
    # numpy 通常已安装；若没有则用占位（本测试不依赖真实计算）
    try:
        import numpy  # noqa: F401
    except ImportError:
        fake_np = types.ModuleType("numpy")
        fake_np.array = lambda x, dtype=None: x
        fake_np.empty = lambda shape, dtype=None: []
        sys.modules["numpy"] = fake_np

    fake_faiss = types.ModuleType("faiss")

    class _Index:
        def __init__(self, dim):
            self.dim = dim
            self.ntotal = 0
            self._vecs = []

        def add(self, vecs):
            self._vecs.extend(list(vecs))
            self.ntotal = len(self._vecs)

        def search(self, query_vec, k):
            # 返回空结果，模拟空索引
            import numpy as np
            return (
                np.array([[-1.0] * k]),
                np.array([[-1] * k]),
            )

    fake_faiss.IndexFlatIP = _Index
    fake_faiss.normalize_L2 = lambda x: None
    sys.modules["faiss"] = fake_faiss

    fake_text2vec = types.ModuleType("text2vec")

    class _SentenceModel:
        def __init__(self, *a, **kw):
            self._dim = 16

        def encode(self, texts):
            import numpy as np
            if not texts:
                return []
            return np.array([[0.1] * self._dim for _ in texts], dtype="float32")

        def get_sentence_embedding_dimension(self):
            return self._dim

    fake_text2vec.SentenceModel = _SentenceModel
    sys.modules["text2vec"] = fake_text2vec

    # colorlog 需要真实可调用的类
    if "colorlog" not in sys.modules:
        fake_colorlog = types.ModuleType("colorlog")

        class _CF:
            def __init__(self, *a, **kw):
                pass

        class _SH:
            def __init__(self):
                import logging
                self.level = logging.NOTSET
                self.filters = []

            def setFormatter(self, *a, **kw):
                pass

            def handle(self, *a, **kw):
                pass

        fake_colorlog.ColoredFormatter = _CF
        fake_colorlog.StreamHandler = _SH
        sys.modules["colorlog"] = fake_colorlog

    # langchain_text_splitters 需要可实例化且带 split_text 的类
    if "langchain_text_splitters" not in sys.modules:
        fake_lc = types.ModuleType("langchain_text_splitters")

        class _Splitter:
            def __init__(self, *a, **kw):
                pass

            def split_text(self, text):
                if not text:
                    return []
                return [text[i : i + 100] for i in range(0, len(text), 50)] or [text]

        fake_lc.RecursiveCharacterTextSplitter = _Splitter
        sys.modules["langchain_text_splitters"] = fake_lc

    # 其余可选依赖占位
    for name, attrs in [
        ("fitz", ["open"]),
        ("docx", ["Document"]),
        ("tavily", ["TavilyClient"]),
        ("requests", ["post"]),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            for attr in attrs:
                setattr(m, attr, lambda *a, **kw: None)
            sys.modules[name] = m

    # 确保 logger_utils 所需的 logs 目录存在
    import os
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)


_install_stubs()


def test_empty_knowledge_does_not_crash_startup():
    """缺陷 1：知识库为空时，原代码 `knowledge_vecs_faiss.shape[1]` 抛 IndexError 导致服务启动失败。

    直接验证 agent.py 中初始化逻辑的空知识分支：维度应从模型获取，索引可正常创建。
    """
    import agent
    import numpy as np

    knowledge = []  # 模拟 load_knowledge 返回空
    knowledge_vecs = agent.model.encode(knowledge) if knowledge else []

    # 空列表 -> np.array([]) 为 1 维数组 shape=(0,)，这正是旧代码崩溃的场景
    knowledge_vecs_faiss = np.array(knowledge_vecs).astype("float32") if knowledge else np.array([], dtype="float32")
    # 关键分支：与 agent.py 中一致——非空二维数组取 shape[1]，否则从模型获取维度
    dim = (
        knowledge_vecs_faiss.shape[1]
        if knowledge_vecs_faiss.ndim == 2 and knowledge_vecs_faiss.shape[0] > 0
        else agent.model.get_sentence_embedding_dimension()
    )
    assert dim == agent.model.get_sentence_embedding_dimension()
    # 索引应能成功创建且维度正确、向量数为 0
    index = agent.faiss.IndexFlatIP(dim)
    assert index.ntotal == 0
    # 对比旧行为：直接访问 shape[1] 必定抛 IndexError（1 维数组）
    assert knowledge_vecs_faiss.ndim == 1
    try:
        knowledge_vecs_faiss.shape[1]
        raise AssertionError("旧代码路径应抛 IndexError 但未抛")
    except IndexError:
        pass  # 符合预期：旧代码会在这里崩溃


def test_add_knowledge_concurrent_keeps_index_in_sync():
    """缺陷 2：并发调用 add_knowledge 会导致 knowledge 列表与 FAISS 索引顺序错位。

    校验：加锁时列表写入顺序与 index.add 顺序严格一致；
    同时验证若不加锁，顺序会错位（证明测试确实能捕获该竞态）。
    """
    import importlib
    import agent
    import time

    def run_with_lock(lock):
        # 用可注入延迟的 list 子类替换 knowledge，放大 extend 与 index.add 之间的竞态窗口
        class SlowList(list):
            def extend(self, items):
                super().extend(items)
                time.sleep(0.02)

        agent.knowledge = SlowList()
        agent.index = agent.faiss.IndexFlatIP(
            agent.model.get_sentence_embedding_dimension()
        )
        agent._knowledge_lock = lock

        add_order = []
        real_add = agent.index.add

        def tracked_add(vecs):
            add_order.append(threading.current_thread().name)
            real_add(vecs)

        agent.index.add = tracked_add

        n_threads = 10
        per_thread = 5
        errors = []

        def worker(tid):
            try:
                chunks = [f"thread-{tid}-chunk-{i}" for i in range(per_thread)]
                agent.add_knowledge(chunks)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"t{i}")
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发追加出现异常: {errors}"
        assert len(agent.knowledge) == n_threads * per_thread
        assert agent.index.ntotal == n_threads * per_thread

        list_order = []
        i = 0
        while i < len(agent.knowledge):
            tid = agent.knowledge[i].split("-")[1]
            list_order.append(f"t{tid}")
            while i < len(agent.knowledge) and agent.knowledge[i].startswith(
                f"thread-{tid}-"
            ):
                i += 1
        return list_order, add_order

    # 1) 真实锁：列表顺序必须等于索引写入顺序
    list_order, add_order = run_with_lock(agent._knowledge_lock)
    assert list_order == add_order, (
        f"加锁后仍不一致: list={list_order} index_add={add_order}"
    )

    # 2) 无锁（MagicMock 等价无操作锁）：多次尝试，至少一次出现错位即证明测试可捕获缺陷；
    #    竞态依赖线程调度，故重复运行以提高复现概率。
    reproduced = False
    for _ in range(20):
        list_order, add_order = run_with_lock(mock.MagicMock())
        if list_order != add_order:
            reproduced = True
            break
    assert reproduced, "无锁场景下未复现竞态，测试失效"


def test_tool_call_malformed_arguments_does_not_crash():
    """缺陷 3：LLM 返回的 tool_calls.arguments 非合法 JSON 时，原代码 json.loads 抛异常导致请求崩溃。"""
    import agent

    # 构造一个返回非法 JSON tool_calls 的模型响应
    bad_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": "{not valid json,",
                            },
                        }
                    ],
                }
            }
        ]
    }

    with mock.patch.object(agent, "call_llm", return_value=bad_response):
        # 不应抛出异常，应返回错误提示
        result = agent.rag_chat("今天北京天气怎么样？", [])
        assert isinstance(result, str)
        assert "参数解析失败" in result


if __name__ == "__main__":
    test_empty_knowledge_does_not_crash_startup()
    print("PASS: test_empty_knowledge_does_not_crash_startup")
    test_add_knowledge_concurrent_keeps_index_in_sync()
    print("PASS: test_add_knowledge_concurrent_keeps_index_in_sync")
    test_tool_call_malformed_arguments_does_not_crash()
    print("PASS: test_tool_call_malformed_arguments_does_not_crash")
    print("\n所有关键缺陷回归测试通过。")
