import time
import faiss
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agent import rag_chat, split_document, add_knowledge, parse_file, get_all_knowledge
from logger_utils import logger

app = FastAPI()


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
        file_bytes = await file.read()
        text = parse_file(file.filename, file_bytes)
        chunks = split_document(text, chunk_size=400)

        added, total = add_knowledge(chunks)

        return {
            "message": "上传成功",
            "filename": file.filename,
            "chunks_added": added,
            "total_chunks": total,
        }

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
