"""
针对 agent.py 中已修复关键缺陷的回归测试。

覆盖的缺陷：
1. 模型返回的 tool_call arguments 为非法 JSON 时，rag_chat 不再抛出未捕获异常
   （修复前会导致 /chat 端点 500 崩溃）。
2. 并发执行 add_knowledge（写）与 search_mostk（读）时，FAISS 索引与 knowledge
   列表受锁保护，不会崩溃或返回错位内容。
3. 知识库文件缺失/为空时，模块导入不再因 np.array([]).shape[1] 抛 IndexError。
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import agent


# ---------------------------------------------------------------------------
# Bug 1: 模型返回非法 JSON 的工具参数不应导致崩溃
# ---------------------------------------------------------------------------

def _tool_call_response(arguments: str) -> dict:
    """构造一个包含 tool_calls 的 DeepSeek 风格响应。"""
    return {
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
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ]
    }


def test_rag_chat_handles_malformed_tool_arguments(monkeypatch):
    """模型返回非法 JSON 字符串时，应返回友好提示而非抛出异常。"""
    # 让模型直接返回一个 tool_call，其 arguments 不是合法 JSON
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda *a, **kw: _tool_call_response("{query: 北京天气,}"),  # 非法 JSON
    )

    # 修复前此处会抛 json.JSONDecodeError
    result = agent.rag_chat("北京今天天气怎么样", [])

    assert isinstance(result, str)
    assert "参数格式错误" in result


def test_rag_chat_handles_null_tool_arguments(monkeypatch):
    """模型返回 arguments 为 None 时不应抛出 TypeError。"""
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda *a, **kw: _tool_call_response(None),
    )

    result = agent.rag_chat("查询一下", [])

    assert isinstance(result, str)
    assert "参数格式错误" in result


def test_rag_chat_still_works_with_valid_tool_arguments(monkeypatch):
    """合法 JSON 参数时，工具调用流程应正常执行（回归保护）。"""
    call_count = {"n": 0}

    def fake_call_llm(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # 第一轮：请求调用 search_web
            return _tool_call_response(json.dumps({"query": "重庆天气"}))
        # 第二轮：直接回答
        return {"choices": [{"message": {"role": "assistant", "content": "重庆今天晴。"}}]}

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "web_search_tavily", lambda q: f"搜索结果：{q}")

    result = agent.rag_chat("重庆天气怎么样", [])

    assert result == "重庆今天晴。"


# ---------------------------------------------------------------------------
# Bug 2: 知识库读写并发安全
# ---------------------------------------------------------------------------

def test_concurrent_add_and_search_does_not_crash():
    """并发 add_knowledge 与 search_mostk 不应崩溃，且 knowledge 与 index 必须保持同步。

    通过给 index.add 注入微小延时来放大竞态窗口：若没有锁保护，knowledge.extend
    与 index.add 会交错执行，导致索引位置与文本块错位，搜索返回错误内容。
    """
    import time as _time

    original_knowledge = list(agent.knowledge)
    errors = []

    # 给 FAISS 的 add 注入随机延时，放大 knowledge.extend 与 index.add 之间的竞态窗口。
    # 随机延时使 index.add 的完成顺序可能不同于 knowledge.extend 的顺序，
    # 若无锁保护，knowledge 与 index 将错位。
    real_add = agent.index.add
    rng = np.random.RandomState(42)

    def slow_add(vecs):
        _time.sleep(rng.uniform(0.001, 0.02))
        real_add(vecs)

    agent.index.add = slow_add
    try:
        added_chunks = []
        lock = threading.Lock()

        def writer(n):
            try:
                for i in range(15):
                    chunk = f"并发块-{n}-{i}-{threading.get_ident()}"
                    agent.add_knowledge([chunk])
                    with lock:
                        added_chunks.append(chunk)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                for _ in range(30):
                    q = agent.model.encode("测试查询")
                    matched = agent.search_mostk(q, K=16)
                    for chunk in matched:
                        assert chunk in agent.knowledge
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # 关键断言：knowledge 与 index 必须严格同步（长度一致）
        assert len(agent.knowledge) == agent.index.ntotal, (
            f"knowledge({len(agent.knowledge)}) 与 index({agent.index.ntotal}) 长度不一致"
        )

        # 关键断言：每个已添加的块都能通过自身向量被正确检索到（顺序未错位）
        for chunk in added_chunks:
            q = agent.model.encode(chunk)
            matched = agent.search_mostk(q, K=1)
            assert matched and matched[0] == chunk, (
                f"检索错位：查询 '{chunk}' 却返回 '{matched[0] if matched else None}'"
            )
    finally:
        agent.index.add = real_add
        # 恢复知识库原状，避免影响其他测试
        with agent._knowledge_lock:
            agent.knowledge[:] = original_knowledge
            agent.index.reset()
            if original_knowledge:
                vecs = agent.model.encode(original_knowledge)
                vecs_faiss = np.array(vecs).astype("float32")
                agent.faiss.normalize_L2(vecs_faiss)
                agent.index.add(vecs_faiss)

    assert not errors, f"并发过程中出现异常: {errors}"


def test_knowledge_lock_is_reentrant():
    """用于保护共享状态的锁必须是可重入锁（同一线程可多次获取）。"""
    assert isinstance(agent._knowledge_lock, type(threading.RLock()))
    with agent._knowledge_lock:
        # 同一线程再次获取不应死锁
        with agent._knowledge_lock:
            pass


# ---------------------------------------------------------------------------
# Bug 3: 知识库为空时模块导入不应崩溃
# ---------------------------------------------------------------------------

def test_import_with_empty_knowledge_base():
    """当知识库文件为空时，导入 agent 不应抛出 IndexError。

    通过子进程实现：在导入 agent 之前把 builtins.open 替换为对知识库文件
    返回空内容，从而触发模块级 ``if knowledge:`` 的 else 分支（Bug 3 的修复点）。
    """
    script = (
        "import sys, types, builtins\n"
        "import numpy as np\n"
        "class Fake:\n"
        "    DIM=8\n"
        "    def encode(self, texts):\n"
        "        single = isinstance(texts, str)\n"
        "        if single: texts=[texts]\n"
        "        vecs=[]\n"
        "        for t in texts:\n"
        "            rng=np.random.RandomState(hash(t)%(2**32))\n"
        "            vecs.append(rng.rand(self.DIM).astype('float32'))\n"
        "        result = np.array(vecs) if vecs else np.array([]).reshape(0,self.DIM)\n"
        "        return result[0] if single else result\n"
        "m=types.ModuleType('text2vec'); m.SentenceModel=Fake; sys.modules['text2vec']=m\n"
        "# 让知识库文件返回空内容，触发空知识库分支\n"
        "_real_open = builtins.open\n"
        "def _patched_open(path, *a, **k):\n"
        "    if '知识库简易版' in str(path):\n"
        "        from io import StringIO\n"
        "        return StringIO('')\n"
        "    return _real_open(path, *a, **k)\n"
        "builtins.open = _patched_open\n"
        "import agent\n"
        "print('EMPTY_KB_IMPORT_OK count=', len(agent.knowledge), 'ntotal=', agent.index.ntotal)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "DEEPSEEK_API_KEY": "dummy", "TAVILY_API_KEY": "dummy"},
        cwd="/workspace",
    )

    assert result.returncode == 0, (
        f"空知识库导入失败:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "EMPTY_KB_IMPORT_OK count= 0" in result.stdout
