"""针对提交后修复的回归测试。

覆盖以下缺陷修复：
1. calculate 函数拒绝 ** 幂运算符，防止内存耗尽型 DoS
2. search_mostk 的 logger.info 移到循环外，避免每次检索重复日志
3. rag_chat 的 tool_summary 使用轮次序号，而非工具名
"""
from unittest.mock import patch, MagicMock

import agent


# ============================================================
# 1. calculate: ** 幂运算符 DoS 防护
# ============================================================

class TestCalculateRejectsPowerOperator:
    """验证 ** 幂运算符被拒绝，防止 9**9**9**9 类表达式耗尽内存。"""

    def test_rejects_simple_power(self):
        result = agent.calculate("9**9")
        assert "错误" in result
        assert "幂运算" in result

    def test_rejects_chained_power(self):
        # 9**9**9**9 会导致服务器挂起/OOM
        result = agent.calculate("9**9**9**9")
        assert "错误" in result
        assert "幂运算" in result

    def test_rejects_large_exponent(self):
        # 9**99999999 会产生约 9500 万位数字
        result = agent.calculate("9**99999999")
        assert "错误" in result
        assert "幂运算" in result

    def test_rejects_power_with_spaces(self):
        result = agent.calculate("2 ** 10")
        assert "错误" in result


class TestCalculateNormalArithmetic:
    """验证修复后普通算术仍然正常工作。"""

    def test_addition(self):
        assert "100" in agent.calculate("40 + 60")

    def test_subtraction(self):
        assert "30" in agent.calculate("50 - 20")

    def test_multiplication(self):
        assert "200" in agent.calculate("50 * 4")

    def test_division(self):
        assert "2.5" in agent.calculate("10 / 4")

    def test_modulo(self):
        assert "1" in agent.calculate("10 % 3")

    def test_parentheses(self):
        assert "14" in agent.calculate("(2 + 5) * 2")

    def test_rejects_invalid_chars(self):
        result = agent.calculate("abc")
        assert "错误" in result
        assert "不支持" in result

    def test_rejects_function_call(self):
        result = agent.calculate("open('x')")
        assert "错误" in result


# ============================================================
# 2. search_mostk: logger.info 应在循环外只调用一次
# ============================================================

class TestSearchMostkLogging:
    """验证 '检索完成' 日志每次查询只记录一次，而非每条结果重复记录。"""

    def test_info_logged_once_per_query(self):
        agent.knowledge = ["块1", "块2", "块3"]
        agent.index.search = MagicMock(
            return_value=([[0.9, 0.3]], [[0, 1]])
        )
        agent.logger.info.reset_mock()

        matched = agent.search_mostk([0.0] * 768, K=2)

        # 修复后：logger.info 只应在循环结束后调用一次
        assert agent.logger.info.call_count == 1
        logged_msg = agent.logger.info.call_args[0][0]
        assert "检索完成" in logged_msg
        assert f"命中{len(matched)}条" in logged_msg

    def test_info_logged_once_even_with_many_results(self):
        agent.knowledge = ["块1", "块2", "块3", "块4"]
        agent.index.search = MagicMock(
            return_value=([[0.9, 0.8, 0.7, 0.1]], [[0, 1, 2, 3]])
        )
        agent.logger.info.reset_mock()

        agent.search_mostk([0.0] * 768, K=4, threshold=0.4)

        # 无论匹配多少条，'检索完成' 只记录一次
        assert agent.logger.info.call_count == 1


# ============================================================
# 3. rag_chat: tool_summary 使用轮次序号
# ============================================================

class TestToolSummaryRoundNumber:
    """验证 tool_summary 使用 '第N次搜索' 而非 '第search_web次搜索'。"""

    @patch("agent.web_search_tavily")
    @patch("agent.call_llm")
    @patch("agent.search_mostk")
    @patch("agent.model")
    def test_uses_round_number_not_tool_name(
        self, mock_model, mock_search, mock_call_llm, mock_web_search
    ):
        mock_model.encode.return_value = [0.0] * 768
        mock_search.return_value = ["相关知识内容"]
        mock_web_search.return_value = "搜索结果详情"

        # 第一次 call_llm 返回工具调用，第二次返回最终答案
        tool_call_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query": "test"}',
                        }
                    }]
                }
            }]
        }
        final_response = {
            "choices": [{
                "message": {"content": "最终答案"}
            }]
        }
        mock_call_llm.side_effect = [tool_call_response, final_response]

        result = agent.rag_chat("测试问题", [])

        assert result == "最终答案"

        # 第二次 call_llm 的 prompt 应包含 "第1次搜索"
        second_call_args = mock_call_llm.call_args_list[1]
        prompt = second_call_args[0][0]  # 第一个位置参数 = user_prompt
        assert "第1次搜索" in prompt
        assert "第search_web次搜索" not in prompt
