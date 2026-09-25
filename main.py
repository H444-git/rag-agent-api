import time
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from agent import rag_chat, split_document, add_knowledge, parse_file, get_all_knowledge
from logger_utils import logger

app = FastAPI()

# 🔒 单文件大小上限：30 MB，防止 OOM DoS
MAX_UPLOAD_SIZE = 30 * 1024 * 1024


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Hello,FastAPI!"}


@app.post("/chat")
def chat(request: ChatRequest):
    start = time.time()
    answer = rag_chat(request.question, [])
    cost = round(time.time() - start, 2)
    logger.info(f"请求处理完成,总耗时{cost}s")
    return {"answer": answer}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # 🔒 先检查 content_length，若 FastAPI 未提供则读取后二次校验
        if file.size is not None and file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（{file.size} 字节），上限 {MAX_UPLOAD_SIZE} 字节",
            )

        file_bytes = await file.read()

        # 🔒 二次校验：防止客户端伪造 content-length 或流式上传绕过
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（{len(file_bytes)} 字节），上限 {MAX_UPLOAD_SIZE} 字节",
            )

        text = parse_file(file.filename, file_bytes)
        chunks = split_document(text, chunk_size=400)

        added, total = add_knowledge(chunks)

        return {
            "message": "上传成功",
            "filename": file.filename,
            "chunks_added": added,
            "total_chunks": total,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("文件上传失败:{e}")
        return {"message": "上传失败", "error": str(e)}


@app.get("/knowledge")
def list_knowledge():
    chunks = get_all_knowledge()
    return {
        "total": len(chunks),
        "chunks": chunks,
    }
