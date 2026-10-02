import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from agent import rag_chat, split_document, add_knowledge, parse_file, get_all_knowledge
from logger_utils import logger

app = FastAPI()

# 上传文件大小上限（10MB），防止超大文件导致 OOM
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


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
        # 只读取上限+1字节，若超过上限则拒绝，避免一次性加载大文件到内存
        file_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="文件大小超过10MB限制")

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
        logger.error(f"文件上传失败:{e}")
        return {"message": "上传失败", "error": str(e)}


@app.get("/knowledge")
def list_knowledge():
    chunks = get_all_knowledge()
    return {
        "total": len(chunks),
        "chunks": chunks,
    }
