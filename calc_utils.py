"""安全数学表达式求值。

从 agent.py 中抽出 calculate，使其不依赖 numpy/faiss/text2vec 等重组件，
便于独立单元测试。求值基于 AST 白名单，禁止属性访问、调用、名称引用等；
幂运算限制指数与结果位数，避免 ``10**10**10`` 这类指数爆炸在 eval 内部
分配巨量内存导致的拒绝服务。
"""

import ast
import math
import operator as _op

# calculate 允许的二元运算符（与字符白名单 0123456789+-*/().% 对应）
_CALC_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
}
_CALC_UNARYOPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}

# 结果十进制位数上限：既避免 10**10**10 这类指数爆炸导致内存耗尽（DoS），
# 也低于 CPython int->str 默认 4300 位上限，避免触发转换异常。
_CALC_MAX_RESULT_DIGITS = 1000

# 表达式最大长度，防止超长表达式本身的解析开销
_CALC_MAX_EXPR_LEN = 200


def _calc_pow(base, exponent):
    """安全计算幂运算，限制指数与结果位数，避免内存耗尽 DoS。"""
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise ValueError("指数必须为整数")
    if exponent < 0:
        # 负指数走浮点路径，结果为定长浮点数，不会分配巨大整数
        return base ** exponent
    if exponent == 0:
        return 1 if isinstance(base, int) else 1.0
    abs_base = abs(base)
    if abs_base <= 1:
        # 0、1、-1 的幂恒为小整数/浮点数，安全
        return base ** exponent
    # 估算结果十进制位数：exponent * log10(|base|)，无需真正计算大整数
    est_digits = exponent * math.log10(abs_base)
    if est_digits > _CALC_MAX_RESULT_DIGITS:
        raise ValueError("结果过大，已拒绝计算")
    return base ** exponent


def _calc_eval(node):
    """递归求值 AST 节点，仅允许数字常量与基本算术运算符。"""
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("仅支持数字常量")
        return node.value
    if isinstance(node, ast.Num):  # 兼容旧版本 Python
        return node.n
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return _calc_pow(_calc_eval(node.left), _calc_eval(node.right))
        if type(node.op) in _CALC_BINOPS:
            return _CALC_BINOPS[type(node.op)](
                _calc_eval(node.left), _calc_eval(node.right)
            )
        raise ValueError("不支持的运算符")
    if isinstance(node, ast.UnaryOp):
        if type(node.op) in _CALC_UNARYOPS:
            return _CALC_UNARYOPS[type(node.op)](_calc_eval(node.operand))
        raise ValueError("不支持的一元运算符")
    raise ValueError("不支持的表达式")


def calculate(expression):
    """
    安全地计算数学表达式。

    使用 AST 白名单求值，禁止属性访问/函数调用/名称引用；
    幂运算限制指数与结果位数，避免 ``10**10**10`` 等导致内存耗尽的 DoS。
    """
    if not isinstance(expression, str) or not expression:
        return "错误：表达式为空"
    if len(expression) > _CALC_MAX_EXPR_LEN:
        return "错误：表达式过长"

    # 字符白名单：只允许数字、基本运算符、括号、小数点与空格
    allowed_chars = set("0123456789+-*/().% ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含不支持的字符"

    try:
        tree = ast.parse(expression, mode="eval")
        result = _calc_eval(tree)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"
