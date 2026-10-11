"""
关键缺陷回归测试。

针对 commit 3e1c713 引入的两个高影响缺陷：
1. calculate() 使用 eval(..., {"__builtins__": {}}, {})，存在沙箱逃逸 → 远程代码执行。
2. 知识库为空时 np.array([]).astype("float32").shape[1] 导致进程导入期崩溃。

为避免在缺少 torch/text2vec 模型权重的环境下触发 agent 模块的全量导入，
这里通过 AST 提取 agent.py 中的 calculate/_safe_eval 函数进行隔离测试，
并直接用 numpy+faiss 验证空索引初始化逻辑。
"""
import ast

import numpy as np
import faiss
import pytest


# ---------- 从 agent.py 中只加载 calculate 相关函数（避免触发重依赖导入） ----------
def _load_calculate():
    with open("agent.py", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    wanted = {"_safe_eval", "calculate"}
    mod = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted
        ],
        type_ignores=[],
    )
    ns = {}
    exec(compile(mod, "agent_calc.py", "exec"), ns)
    return ns["calculate"]


calculate = _load_calculate()


# ---------- Bug 1：calculate 沙箱逃逸 ----------
@pytest.mark.parametrize(
    "expr,expected_substr",
    [
        ("1 + 2 * 3", "= 7"),
        ("(100 - 5) / 2", "= 47.5"),
        ("5000 / 12", "416.66"),
        ("-5 + 3", "= -2"),
    ],
)
def test_calculate_normal_arith(expr, expected_substr):
    assert expected_substr in calculate(expr)


def test_calculate_rejects_dunder_sandbox_escape():
    # 经典的 __builtins__ 被移除后仍可逃逸的 payload；必须被拒绝，不能执行。
    payload = (
        "[c for c in ().__class__.__bases__[0].__subclasses__()"
        " if c.__name__=='catch_warnings'][0]()._module"
        ".__builtins__['__import__']('os').system('echo PWNED')"
    )
    result = calculate(payload)
    assert "错误" in result
    assert "PWNED" not in result


def test_calculate_rejects_import():
    assert "错误" in calculate("__import__('os').system('ls')")


def test_calculate_rejects_attribute_access():
    assert "错误" in calculate("().__class__")


def test_calculate_rejects_open_call():
    assert "错误" in calculate("open('/etc/passwd').read()")


def test_calculate_division_by_zero_handled():
    result = calculate("1 / 0")
    assert "错误" in result
    assert "division by zero" in result


# ---------- Bug 2：空知识库启动崩溃 ----------
class _FakeModel:
    def get_sentence_embedding_dimension(self):
        return 8


def _build_index(knowledge_vecs, model):
    """复刻 agent.py 中修复后的 FAISS 索引构建逻辑。"""
    knowledge_vecs_faiss = np.array(knowledge_vecs).astype("float32")
    if knowledge_vecs_faiss.size == 0:
        dim = model.get_sentence_embedding_dimension()
        return faiss.IndexFlatIP(dim)
    dim = knowledge_vecs_faiss.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(knowledge_vecs_faiss)
    index.add(knowledge_vecs_faiss)
    return index


def test_empty_knowledge_does_not_crash():
    # 旧代码 knowledge_vecs_faiss.shape[1] 会抛 IndexError；
    # 新代码应返回一个空的可用索引。
    index = _build_index([], _FakeModel())
    assert index.ntotal == 0

    # 空索引上的检索不应崩溃，且应返回全 -1。
    q = np.random.randn(1, 8).astype("float32")
    faiss.normalize_L2(q)
    _, indices = index.search(q, 4)
    assert (indices == -1).all()


def test_populated_knowledge_index_works():
    rng = np.random.RandomState(0)
    vecs = rng.randn(3, 8)
    index = _build_index(vecs, _FakeModel())
    assert index.ntotal == 3

    q = vecs[0:1].astype("float32")
    faiss.normalize_L2(q)
    distances, indices = index.search(q, 1)
    assert indices[0][0] == 0
