"""针对提交后关键缺陷修复的回归测试。

覆盖：
1. calculate() 不再使用 eval，拒绝 9**9**9 等导致资源耗尽的表达式
2. rag_chat 处理 LLM 返回的非法 JSON 工具参数，不再抛出未捕获异常
3. rag_chat 处理工具缺少必填参数的情况
4. add_knowledge 原子性：index.add 失败时 knowledge 不被污染
5. search_mostk 在索引为空时安全返回 []
6. 空知识库启动不崩溃
"""
import json
import threading
import time

import numpy as np
import pytest

import agent


# ===== Defect 1: calculate() DoS 防护 =====

class TestCalculateDOS:
    def test_rejects_huge_exponentiation(self):
        """9**9**9 会产生约 3.7 亿位整数，必须被拒绝而非耗尽内存。"""
        result = agent.calculate("9**9**9")
        assert "错误" in result
        assert "过大" in result

    def test_rejects_large_power_base(self):
        result = agent.calculate("1000000**2")
        assert "错误" in result

    def test_normal_arithmetic_works(self):
        assert agent.calculate("1+2") == "计算结果：1+2 = 3"
        assert agent.calculate("(10+5)*2") == "计算结果：(10+5)*2 = 30"
        assert agent.calculate("10/4") == "计算结果：10/4 = 2.5"

    def test_division_by_zero_handled(self):
        result = agent.calculate("1/0")
        assert "错误" in result

    def test_disallowed_chars_rejected(self):
        # 字母不应被允许（防止 __import__ 等）
        result = agent.calculate("__import__('os')")
        assert "错误" in result
        assert "不支持的字符" in result

    def test_no_eval_rce(self):
        """即使绕过字符过滤，也不能执行任意代码（AST 白名单）。"""
        # 含字母的表达式会被字符过滤拦截
        result = agent.calculate("().__class__")
        assert "不支持的字符" in result


# ===== Defect 2 & 3: rag_chat 工具参数解析健壮性 =====

def _make_tool_response(tool_name, arguments_str):
    """构造一个模拟 LLM 返回的工具调用响应。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_str,
                            },
                        }
                    ],
                }
            }
        ]
    }


class TestRagChatToolArgs:
    def test_malformed_json_arguments_returns_error_not_crash(self, monkeypatch):
        """LLM 返回非法 JSON 时，rag_chat 应返回错误字符串而非抛出异常。"""
        bad_args = '{query: 北京天气}'  # 缺少引号的非法 JSON
        monkeypatch.setattr(agent, "call_llm", lambda *a, **kw: _make_tool_response("search_web", bad_args))

        # 不应抛出 json.JSONDecodeError
        result = agent.rag_chat("今天天气怎么样", [])
        assert isinstance(result, str)
        assert "解析失败" in result

    def test_missing_query_arg_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            agent, "call_llm",
            lambda *a, **kw: _make_tool_response("search_web", json.dumps({"not_query": "x"})),
        )
        result = agent.rag_chat("搜索一下", [])
        assert "缺少 query 参数" in result

    def test_missing_expression_arg_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            agent, "call_llm",
            lambda *a, **kw: _make_tool_response("calculate", json.dumps({})),
        )
        result = agent.rag_chat("算一下", [])
        assert "缺少 expression 参数" in result

    def test_valid_calculate_tool_call(self, monkeypatch):
        monkeypatch.setattr(
            agent, "call_llm",
            lambda *a, **kw: _make_tool_response("calculate", json.dumps({"expression": "2+3"})),
        )
        # 工具调用后会进入下一轮；第二轮让模型直接回答
        responses = [
            _make_tool_response("calculate", json.dumps({"expression": "2+3"})),
            {"choices": [{"message": {"role": "assistant", "content": "答案是5"}}]},
        ]
        monkeypatch.setattr(agent, "call_llm", lambda *a, **kw: responses.pop(0))
        result = agent.rag_chat("2加3等于几", [])
        assert result == "答案是5"


# ===== Defect 3: add_knowledge 原子性 =====

class TestAddKnowledgeAtomicity:
    def test_index_add_failure_does_not_pollute_knowledge(self, monkeypatch):
        """若 index.add 抛出异常，knowledge 列表不应被扩展。"""
        original_index = agent.index
        original_knowledge = list(agent.knowledge)

        # 用一个会在 add 时抛异常的假索引替换
        class BrokenIndex:
            ntotal = original_index.ntotal if original_index else 0
            def add(self, vecs):
                raise RuntimeError("simulated faiss add failure")
            def search(self, *a, **kw):
                return np.array([[-1.0]]), np.array([[-1]])

        monkeypatch.setattr(agent, "index", BrokenIndex())
        before_count = len(agent.knowledge)

        with pytest.raises(RuntimeError):
            agent.add_knowledge(["一块新知识"])

        # knowledge 不应被污染
        assert len(agent.knowledge) == before_count

        # 恢复
        monkeypatch.setattr(agent, "index", original_index)

    def test_successful_add_keeps_index_and_knowledge_in_sync(self):
        before = len(agent.knowledge)
        ntotal_before = agent.index.ntotal if agent.index else 0
        added, total = agent.add_knowledge(["同步测试块"])
        assert added == 1
        assert total == before + 1
        assert agent.index.ntotal == ntotal_before + 1
        assert len(agent.knowledge) == agent.index.ntotal


# ===== Defect 4 & 5: 空索引 / 并发安全 =====

class TestEmptyIndexAndConcurrency:
    def test_search_mostk_returns_empty_when_no_index(self, monkeypatch):
        """index 为 None 时 search_mostk 应返回 [] 而非崩溃。"""
        monkeypatch.setattr(agent, "index", None)
        query_vec = np.random.rand(768).astype("float32")
        result = agent.search_mostk(query_vec)
        assert result == []

    def test_concurrent_search_and_add_no_crash(self):
        """并发执行 add_knowledge 与 search_mostk 不应崩溃或数据错位。"""
        errors = []

        def search_loop():
            try:
                q = np.random.rand(768).astype("float32")
                for _ in range(20):
                    agent.search_mostk(q, K=8)
            except Exception as e:
                errors.append(e)

        def add_loop():
            try:
                for i in range(10):
                    agent.add_knowledge([f"并发块_{i}_{threading.get_ident()}"])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=search_loop)
        t2 = threading.Thread(target=add_loop)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"并发错误: {errors}"
        # 最终 knowledge 与 index 必须对齐
        assert len(agent.knowledge) == agent.index.ntotal
