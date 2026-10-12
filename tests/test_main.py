import sys
from unittest.mock import MagicMock

import pytest

# 在导入 main 之前 mock agent，避免拉入真实的模型与 FAISS 初始化
_mock_agent = MagicMock()
_mock_agent.rag_chat.return_value = "test answer"
_mock_agent.split_document.return_value = ["chunk1", "chunk2"]
_mock_agent.add_knowledge.return_value = (2, 5)
_mock_agent.parse_file.return_value = "parsed text"
_mock_agent.get_all_knowledge.return_value = ["chunk1", "chunk2"]
sys.modules["agent"] = _mock_agent

sys.modules["logger_utils"] = MagicMock()
sys.modules["logger_utils"].logger = MagicMock()

from fastapi.testclient import TestClient
from main import app, MAX_UPLOAD_SIZE

client = TestClient(app)


def test_upload_rejects_oversized_file():
    """上传超过大小限制的文件应被拒绝，防止内存耗尽。"""
    large_content = b"x" * (MAX_UPLOAD_SIZE + 1)
    response = client.post(
        "/upload",
        files={"file": ("large.txt", large_content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "上传失败"
    assert "超过限制" in data["error"]


def test_upload_accepts_normal_file():
    """正常大小的文件应走通解析与入库流程。"""
    content = b"this is a small test file"
    response = client.post(
        "/upload",
        files={"file": ("small.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "上传成功"
    assert data["chunks_added"] == 2
    assert data["total_chunks"] == 5
