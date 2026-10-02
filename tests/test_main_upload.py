import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """构造一个使用模拟 agent 模块的 TestClient，避免加载真实模型。"""
    fake_agent = MagicMock()
    fake_agent.rag_chat = MagicMock()
    fake_agent.split_document = MagicMock(return_value=["chunk1"])
    fake_agent.add_knowledge = MagicMock(return_value=(1, 1))
    fake_agent.parse_file = MagicMock(return_value="text")
    fake_agent.get_all_knowledge = MagicMock(return_value=[])

    sys.modules["agent"] = fake_agent
    sys.modules.pop("main", None)

    from main import app

    return TestClient(app)


def test_upload_rejects_oversized_file(client):
    """上传超过 10MB 的文件应返回 413，避免一次性读入大文件导致 OOM。"""
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/upload",
        files={"file": ("test.txt", large_content, "text/plain")},
    )

    assert response.status_code == 413
    assert "10MB" in response.json()["detail"]


def test_upload_accepts_normal_file(client):
    """正常大小的文件应能成功上传并更新知识库。"""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"small content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "上传成功"
    assert response.json()["chunks_added"] == 1
