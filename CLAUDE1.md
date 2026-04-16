# CLAUDE.md — 多模态 RAG Agent

## 项目概述

企业级**多模态 RAG（检索增强生成）**系统，用于智能文档问答。系统接收 PDF 及结构化文档，通过混合检索（稠密 + 稀疏）建立索引，最终由基于 Qwen 模型的 LangGraph ReAct Agent 生成答案。

**核心能力：**
- 多模态文档解析：文本、表格、图像、公式
- 混合检索：稠密（Chroma 向量）+ 稀疏（BM25），RRF 融合 + 可选精排
- LangGraph ReAct Agent，配备扩展工具（计算器、网络搜索、表格分析、图像描述）
- 流式 SSE API 与 Streamlit 可视化前端
- 离线评估框架（Recall / MRR / Precision / NDCG）

---

## 系统架构

```
PDF / 文档
    │
    ▼
MinerU OCR (scripts/mineru_ocr.py)   ← 预处理，PDF → 结构化 Markdown
    │
    ▼
DocumentRouter (src/document_parser/router.py)
    ├─ TextParser  → ParsedBlock[]（HEADER, TABLE, FIGURE, FORMULA, TEXT）
    └─ VisionParser → ParsedBlock[]（Qwen-VL 图像理解）
    │
    ▼
StructureAwareChunker (src/document_parser/chunker.py)
    │  标题 → 上下文前缀；表格/图注 → 原子块；文本 → 滑动窗口切分
    ▼
ChromaVectorStore (src/vectorstore/chroma_store.py)
    ├─ 稠密索引：Qwen3-Embedding-0.6B（1024 维）
    └─ 稀疏索引：BM25（jieba 中文分词）
    │
    ▼
HybridRetriever (src/retrieval/hybrid_retriever.py)
    ├─ DenseRetriever (src/retrieval/dense_retriever.py)
    ├─ SparseRetriever (src/retrieval/sparse_retriever.py)
    ├─ RRF 融合：score(d) = Σ 1/(k=60 + rank_i(d))
    └─ QwenReranker (src/retrieval/reranker.py) — 可选 cross-encoder 精排
    │
    ▼
MultimodalRAGAgent (src/agent/rag_agent.py)   ← LangGraph ReAct
    ├─ 工具：knowledge_base_search, query_rewrite, multi_round_search
    └─ 技能：calculator, web_search, table_analyzer, image_describer, summarizer
    │
    ├─ FastAPI (src/api/main.py)    ← REST + SSE 流式接口
    └─ Streamlit UI (frontend/app.py)
```

---

## 开发命令

### 环境安装
```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 DASHSCOPE_API_KEY
```

### 文档入库流程
```bash
# 第一步：OCR 预处理（PDF → 结构化 Markdown）
python scripts/mineru_ocr.py --input-dir ./data/documents

# 第二步：批量入库 OCR 输出
python scripts/ingest_ocr_output.py --chunk-strategy semantic

# 或：直接入库指定文档
python scripts/ingest_documents.py --file ./data/my_doc.md
```

### 启动服务
```bash
# 后端 API（开发模式，热重载）
python run_server.py --reload

# 后端 API（生产模式）
python run_server.py --host 0.0.0.0 --port 8000 --workers 4

# 前端 UI
streamlit run frontend/app.py
# → http://localhost:8501
```

### 运行测试
```bash
pytest tests/ -v
pytest tests/test_parser.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_agent.py -v
```

### 运行评估
```bash
# 离线模式（仅 BM25，无需 API Key）
python scripts/evaluate.py --dataset ./data/eval_dataset.jsonl

# 在线模式（完整流水线：Dense + Sparse + Reranker）
python scripts/evaluate.py --online

# Chunk 策略消融实验
python scripts/evaluate.py --chunk-ablation all
```

### API 调用示例
```bash
# 同步问答
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "公司2023年收入是多少?"}'

# 流式问答（SSE）
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍主要业务"}'

# 上传文档
curl -X POST http://localhost:8000/v1/ingest/upload \
  -F "files=@document.md"
```

---

## 配置说明

所有配置集中在 [config/settings.py](config/settings.py)，基于 Pydantic-settings。可通过 `.env` 文件或环境变量覆盖。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | （必填） | Qwen 模型的 DashScope API Key |
| `LLM_MODE` | `api` | `api`（DashScope）或 `local`（本地部署） |
| `LLM_MODEL` | `qwen3.6-plus` | API 模式主推理模型 |
| `LOCAL_LLM_MODEL` | `qwen3-8b` | 本地模式模型名称 |
| `VL_MODEL` | `qwen-vl-max` | 多模态视觉模型 |
| `EMBEDDING_MODE` | `local` | `local`（本地推理）或 `api` |
| `LOCAL_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 本地 Embedding 模型路径 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `ENABLE_RERANK` | `True` | 是否启用 cross-encoder 精排 |
| `RERANKER_MODE` | `local` | `local` 或 `api` |
| `CHUNK_STRATEGY` | `semantic` | 切分策略 |
| `CHUNK_SIZE` | `1024` | 近似 token 数 |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `10` | 检索候选数 |
| `FINAL_TOP_K` | `5` | 精排后返回文档数 |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | `0.7` / `0.3` | RRF 融合权重 |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | 向量库持久化路径 |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | 服务绑定地址/端口 |

---

## 关键文件

| 文件 | 用途 |
|---|---|
| [config/settings.py](config/settings.py) | 全局 Pydantic 配置 |
| [src/document_parser/base_parser.py](src/document_parser/base_parser.py) | `ParsedBlock` 数据结构 + 抽象基类 |
| [src/document_parser/text_parser.py](src/document_parser/text_parser.py) | Markdown/TXT 结构化解析（标题、表格、公式） |
| [src/document_parser/chunker.py](src/document_parser/chunker.py) | 6 种切分策略，默认 `semantic` |
| [src/embeddings/qwen_embedding.py](src/embeddings/qwen_embedding.py) | Qwen3-Embedding 双模式封装 |
| [src/vectorstore/chroma_store.py](src/vectorstore/chroma_store.py) | Chroma 持久化向量库 + CRUD |
| [src/retrieval/hybrid_retriever.py](src/retrieval/hybrid_retriever.py) | RRF 融合 + 精排编排 |
| [src/retrieval/sparse_retriever.py](src/retrieval/sparse_retriever.py) | BM25 稀疏检索（jieba 分词） |
| [src/retrieval/reranker.py](src/retrieval/reranker.py) | Qwen3-Reranker cross-encoder |
| [src/agent/rag_agent.py](src/agent/rag_agent.py) | LangGraph ReAct Agent（同步 + 流式） |
| [src/agent/tools.py](src/agent/tools.py) | 检索工具定义 |
| [src/agent/skills/](src/agent/skills/) | 扩展技能：计算器、网络搜索等 |
| [src/api/main.py](src/api/main.py) | FastAPI 接口（REST + SSE） |
| [src/api/schemas.py](src/api/schemas.py) | Pydantic 请求/响应模型 |
| [frontend/app.py](frontend/app.py) | Streamlit 多模块前端 |
| [scripts/mineru_ocr.py](scripts/mineru_ocr.py) | MinerU OCR 预处理 |
| [scripts/ingest_ocr_output.py](scripts/ingest_ocr_output.py) | OCR 结果批量入库 CLI |
| [scripts/evaluate.py](scripts/evaluate.py) | 检索评估（Recall/MRR/NDCG） |
| [run_server.py](run_server.py) | FastAPI 服务启动入口 |

---

## 代码规范

### 文档解析
- `ParsedBlock` 是核心数据单元，包含 `block_type`、`content`、`metadata`，以及可选的 `header_context`
- Block 类型：`HEADER`、`TEXT`、`TABLE`、`FIGURE`、`FORMULA`
- 表格和图注始终作为**原子块**处理，不跨 chunk 截断
- 标题存储为后续文本块的 `header_context`，不单独输出为 chunk

### 检索流水线
- `DenseRetriever` 将 L2 距离归一化为 [0,1] 分数（越高越相似）
- `SparseRetriever` 在每次 `add_documents()` 调用时重建 BM25 索引
- RRF 公式：`score(d) = Σ 1/(60 + rank_i(d))`，k=60 硬编码于 `hybrid_retriever.py`
- 精排输入大小为 `rerank_top_k`，最终输出为 `final_top_k`

### Agent / 工具
- Agent 使用 `langgraph` 的 `create_react_agent`，LLM 为 `langchain_community.chat_models.ChatTongyi`
- 扩展技能按需注册，各技能依赖详见 `src/agent/skills/` 对应文件
- 流式响应推送 SSE 事件：`tool_start`、`tool_done`、`answer`

### 配置访问
- 始终通过 `from config.settings import settings` 访问配置（单例）
- 配置项支持同名大写环境变量覆盖

### 日志
- 全项目使用 `loguru`：`from loguru import logger`
- 不使用标准库 `logging` 模块

---

## 数据目录

| 路径 | 内容 |
|---|---|
| `data/pdfs/` | 原始 PDF 文件 |
| `data/ocr_output/` | MinerU OCR 输出（Markdown + 图片） |
| `data/chroma_db/` | Chroma 向量库持久化数据 |
| `data/eval_corpus/` | 评估用文档语料 |
| `data/eval_dataset*.jsonl` | 问答评估数据集 |

---

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/query` | 同步问答（支持对话历史） |
| `POST` | `/v1/query/stream` | SSE 流式问答 |
| `POST` | `/v1/ingest/upload` | 上传并入库文件 |
| `GET` | `/v1/documents` | 文档列表（分页） |
| `GET` | `/v1/documents/{id}` | 获取单个文档 |
| `PUT` | `/v1/documents/{id}` | 更新文档 |
| `DELETE` | `/v1/documents/{id}` | 删除文档 |
| `DELETE` | `/v1/collection` | 重置整个集合 |

---

## 模型参考

完整 Qwen API 模型列表见 [docs/qwen_api_models.md](docs/qwen_api_models.md)。

| 角色 | 本地模型 | API 模型 |
|---|---|---|
| LLM | `qwen3-8b` | `qwen3.6-plus` |
| 视觉 | — | `qwen-vl-max` |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | `text-embedding-v4` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | `qwen3-rerank` |

---

## 评估基准（在线混合检索 + 精排，173 条查询）

| 指标 | 无精排 | 有精排 |
|---|---|---|
| Recall@1 | 76.30% | **86.71%** |
| MRR@1 | 0.7630 | **0.8671** |
| Precision@1 | 76.30% | **86.71%** |
| NDCG@1 | 0.7630 | **0.8671** |
| Recall@5 | 94.80% | **95.95%** |
| NDCG@5 | 0.8638 | **0.9223** |
