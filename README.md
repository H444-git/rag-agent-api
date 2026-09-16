# 🤖 RAG Agent API

> 基于 FastAPI + FAISS + DeepSeek 的 RAG Agent 服务，支持知识库检索、多工具调用、文件上传和动态更新。

## 📌 项目简介

这是一个完整的 RAG（检索增强生成）Agent 服务，通过 HTTP API 对外提供智能问答能力。系统在本地构建向量知识库，结合 DeepSeek 大模型和 Tavily 联网搜索，实现“知识库检索 → 工具调用 → 生成回答”的完整链路。

## ✨ 功能特点

- 🔍 **向量检索**：基于 FAISS + text2vec 的语义检索，支持 Top-K 和阈值过滤
- 🧠 **多工具调用**：Agent 可自主决定调用 `search_web`（联网搜索）或 `calculate`（数学计算）
- 📄 **多格式上传**：支持 `.txt`、`.pdf`、`.docx` 文件上传，自动分块并加入知识库
- 🔄 **动态更新**：上传后立即可检索，无需重启服务
- 📊 **结构化日志**：完整记录请求链路、工具调用、检索结果和耗时
- 🌐 **Web 界面**：FastAPI 自带 Swagger 文档，可在 `/docs` 直接测试

## 🛠️ 技术栈

| 模块     | 技术                 |
| -------- | -------------------- |
| Web 框架 | FastAPI + Uvicorn    |
| 向量检索 | FAISS + text2vec     |
| 大模型   | DeepSeek API         |
| 联网搜索 | Tavily API           |
| 文档解析 | PyMuPDF、python-docx |
| 日志     | colorlog             |
| 配置管理 | python-dotenv        |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/H444-git/rag-agent-api.git
cd rag-agent-api
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. 启动服务

```bash
uvicorn main:app --reload
```

服务启动后访问 `http://127.0.0.1:8000/docs` 查看接口文档。

## 📡 API 说明

### `POST /chat` — 智能问答

```json
{
  "question": "RAG是什么？"
}
```

返回：

```json
{
  "answer": "..."
}
```

### `POST /upload` — 上传文档

使用 `multipart/form-data` 上传文件，支持 `.txt`、`.pdf`、`.docx`。

返回：

```json
{
  "message": "上传成功",
  "filename": "example.pdf",
  "chunks_added": 17,
  "total_chunks": 26
}
```

### `GET /knowledge` — 查看知识库

返回当前知识库所有文本块和总数。

## 🏗️ 项目结构

```
rag-agent-api/
├── main.py              # FastAPI 入口，路由定义
├── agent.py             # Agent 核心逻辑（检索、工具调用、生成）
├── logger_utils.py      # 日志配置
├── 知识库简易版.txt      # 初始知识库
├── requirements.txt     # 依赖清单
├── .env.example         # 环境变量示例
└── README.md
```

## 🔄 工作流程

```
用户提问
    │
    ▼
FastAPI /chat 接口
    │
    ▼
FAISS 向量检索（本地知识库）
    │
    ▼
调用 DeepSeek（带 tools）
    │
    ├── 直接回答 → 返回
    │
    └── 调用工具 → 执行 → 再次调用模型 → 返回
```

## 📝 后续计划

- [ ] 流式输出（SSE）
- [ ] 知识库持久化
- [ ] 向量数据库替换（Chroma/FAISS 持久化）
- [ ] Reranker 重排序
- [ ] Docker 部署

## 📄 License

MIT
