"""
提交后正确性检查的回归测试。

覆盖本次修复的三个关键缺陷：
1. calculate() 的 DoS 漏洞（eval + ** 导致大整数 OOM）
2. /upload 与 /chat 并发访问全局 FAISS 索引的竞态条件
3. parse_pdf() 在异常时未关闭 fitz.Document 导致的资源泄漏
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 在导入 agent 之前，mock 掉需要 torch/联网的重型依赖，使测试可独立运行
# ---------------------------------------------------------------------------
def _install_mocks():
    # text2vec 依赖 torch，用一个假的 SentenceModel 替换
    fake_text2vec = types.ModuleType("text2vec")

    class _FakeSentenceModel:
        def encode(self, texts):
            # 返回一个形状为 (n, 8) 的确定性向量，满足 FAISS 建库需求
            n = len(texts) if isinstance(texts, list) else 1
            rng = np.random.default_rng(42)
            return rng.random((n, 8)).astype("float32")

    fake_text2vec.SentenceModel = _FakeSentenceModel
    sys.modules["text2vec"] = fake_text2vec

    # tavily 客户端在导入时实例化，mock 掉
    fake_tavily = types.ModuleType("tavily")
    fake_tavily.TavilyClient = MagicMock()
    sys.modules["tavily"] = fake_tavily


_install_mocks()

import agent  # noqa: E402


# ===========================================================================
# 1. calculate() DoS 漏洞回归测试
# ===========================================================================
class TestCalculateDoS:
    """calculate 不得使用 eval，必须拒绝 ** 与超大数运算。"""

    @pytest.mark.parametrize(
        "expr, expected_substr",
        [
            ("2+2", "计算结果"),
            ("10*5", "计算结果"),
            ("100/4", "计算结果"),
            ("(1+2)*3", "计算结果"),
            ("10%3", "计算结果"),
            ("10//3", "计算结果"),
            ("-5+3", "计算结果"),
            ("3.14*2", "计算结果"),
        ],
    )
    def test_normal_arithmetic_works(self, expr, expected_substr):
        result = agent.calculate(expr)
        assert expected_substr in result

    def test_power_operator_rejected(self):
        """** 必须被拒绝，防止 10**10**10 这类大整数耗尽内存。"""
        result = agent.calculate("10**10**10")
        assert "计算错误" in result
        assert "不支持的运算符" in result

    def test_nested_power_rejected(self):
        result = agent.calculate("2**2**2**2")
        assert "计算错误" in result

    def test_large_multiplication_rejected(self):
        """连乘超过 1e100 的结果必须被拒绝，防止内存耗尽。"""
        big = "*".join(["9999999999"] * 11)  # (1e10)^11 = 1e110 > 1e100
        result = agent.calculate(big)
        assert "计算错误" in result

    def test_import_rejected(self):
        """不得执行任意代码。"""
        result = agent.calculate('__import__("os")')
        assert "计算错误" in result

    def test_statement_rejected(self):
        result = agent.calculate("1+1; import os")
        assert "计算错误" in result

    def test_zero_division_handled(self):
        result = agent.calculate("1/0")
        assert "计算错误" in result

    def test_no_eval_used(self):
        """calculate 函数体内不得出现 eval 调用。"""
        import inspect

        src = inspect.getsource(agent.calculate)
        assert "eval(" not in src


# ===========================================================================
# 2. FAISS 索引并发安全回归测试
# ===========================================================================
class TestKnowledgeConcurrency:
    """
    /upload (async, 事件循环) 调用 add_knowledge 写入，
    /chat (sync, 线程池) 调用 search_mostk 读取，
    两者共享全局 FAISS 索引，必须由 _knowledge_lock 保护。
    """

    def test_lock_exists(self):
        assert hasattr(agent, "_knowledge_lock")

    def test_add_and_search_use_lock(self):
        """add_knowledge 与 search_mostk 必须获取 _knowledge_lock。"""
        import inspect

        add_src = inspect.getsource(agent.add_knowledge)
        search_src = inspect.getsource(agent.search_mostk)
        assert "_knowledge_lock" in add_src
        assert "_knowledge_lock" in search_src

    def test_concurrent_add_search_does_not_crash(self):
        """
        模拟 /upload 与 /chat 并发：一个线程持续 add_knowledge，
        另一个线程持续 search_mostk。在加锁前这段代码会因 FAISS
        非线程安全而偶发 segfault / 返回错误结果；加锁后应稳定运行。
        """
        import threading
        import time

        errors = []

        def writer():
            try:
                for i in range(50):
                    agent.add_knowledge([f"并发测试块 {i}"])
            except Exception as e:  # noqa: BLE001
                errors.append(("writer", repr(e)))

        def reader():
            try:
                rng = np.random.default_rng(0)
                for _ in range(200):
                    q = rng.random(8).astype("float32")
                    agent.search_mostk(q, K=4)
            except Exception as e:  # noqa: BLE001
                errors.append(("reader", repr(e)))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not t1.is_alive(), "writer 线程未结束（可能死锁）"
        assert not t2.is_alive(), "reader 线程未结束（可能死锁）"
        assert errors == [], f"并发期间出现异常: {errors}"

    def test_knowledge_index_stay_consistent(self):
        """add_knowledge 后 knowledge 长度与 index.ntotal 必须一致。"""
        before = len(agent.get_all_knowledge())
        agent.add_knowledge(["一致性测试块"])
        after = len(agent.get_all_knowledge())
        assert after == before + 1
        assert agent.index.ntotal == after


# ===========================================================================
# 3. parse_pdf 资源泄漏回归测试
# ===========================================================================
class TestParsePdfResourceLeak:
    """parse_pdf 在解析异常时也必须关闭 fitz.Document。"""

    def test_uses_context_manager(self):
        import inspect

        src = inspect.getsource(agent.parse_pdf)
        # 必须使用 with fitz.open(...) 而不是手动 doc.close()
        assert "with fitz.open" in src
        assert "doc.close()" not in src

    def test_closes_doc_on_exception(self, monkeypatch):
        """
        模拟 page.get_text() 抛异常，验证 fitz.Document 仍被关闭。
        """
        import fitz

        close_calls = []

        real_open = fitz.open

        class _FakePage:
            def get_text(self):
                raise RuntimeError("模拟解析失败")

        class _FakeDoc:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return iter([_FakePage()])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.closed = True
                close_calls.append(True)
                return False

        def fake_open(*args, **kwargs):
            return _FakeDoc()

        monkeypatch.setattr(fitz, "open", fake_open)

        with pytest.raises(RuntimeError, match="模拟解析失败"):
            agent.parse_pdf(b"%PDF-1.4 fake")

        assert close_calls == [True], "fitz.Document 未被关闭，存在资源泄漏"
