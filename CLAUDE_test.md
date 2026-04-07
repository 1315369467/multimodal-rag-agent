# CLAUDE.md

## 项目概述

多模态 RAG Agent — 基于 Qwen 系列模型 + DashScope API 的企业级知识库问答系统。
支持 PDF（文本/扫描件/图表）、TXT、Markdown 文档的解析、检索与问答。

## 技术栈

- **LLM/Embedding**: Qwen3.5 (qwen3-235b-a22b) + text-embedding-v3 + gte-rerank，通过 DashScope OpenAI 兼容接口调用
- **Agent**: LangGraph ReAct Agent（非 LangChain AgentExecutor）
- **向量库**: ChromaDB（持久化）
- **检索**: 稠密 + BM25 稀疏 + RRF 融合 + Reranker 精排
- **API**: FastAPI + Uvicorn
- **前端**: Streamlit
- **配置**: pydantic-settings + `.env`

## 项目结构

```
config/settings.py        # 全局配置（pydantic-settings 单例）
src/document_parser/      # 文档解析：PDF/TXT/MD → ParsedBlock → SemanticChunker
src/embeddings/           # Qwen Embedding 封装
src/vectorstore/          # ChromaDB CRUD
src/retrieval/            # Dense/Sparse/Hybrid 检索 + Reranker
src/agent/                # LangGraph ReAct Agent + 工具定义
src/api/                  # FastAPI REST 接口
frontend/app.py           # Streamlit 前端
scripts/                  # CLI 工具（批量入库、评估）
tests/                    # pytest 单元测试（44 用例，全部 mock）
```

## 核心数据流

`文档 → DocumentRouter → BaseDocumentParser.parse() → list[ParsedBlock] → SemanticChunker.chunk() → list[Document] → ChromaVectorStore.add_documents()`

`用户问题 → MultimodalRAGAgent.query() → LangGraph ReAct 循环 → HybridRetriever.retrieve() → 答案 + 来源`

## 编码规约

- Python 3.10+，使用 `from __future__ import annotations` 延迟类型注解
- 类型标注用 `X | None` 而非 `Optional[X]`，`list[str]` 而非 `List[str]`
- 数据类用 `@dataclass`，API schema 用 Pydantic `BaseModel`
- 日志统一用 `loguru.logger`，不用 `logging`
- 配置通过 `config.settings.get_settings()` 单例获取，支持 `.env` 覆盖
- 模块内导入用相对路径（`from .base_parser import ...`），跨模块用绝对路径（`from src.retrieval.hybrid_retriever import ...`）
- 文档字符串和注释使用中文
- 代码内分隔线风格：`# ── 段落标题 ──────────────────`

## 测试

```bash
pytest tests/ -v
```

- 测试全部 mock 外部依赖（DashScope API、ChromaDB），无需网络和 API Key
- 测试文件按模块划分：`test_parser.py`、`test_retrieval.py`、`test_agent.py`

## 运行

```bash
# 后端
python run_server.py
# 或
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 前端
streamlit run frontend/app.py
```

## 注意事项

- `.env` 含 `DASHSCOPE_API_KEY`，已在 `.gitignore` 中排除
- Chroma metadata 仅支持标量值，嵌套 dict/list 需通过 `ChromaVectorStore._flatten_metadata()` 扁平化
- Embedding 维度为 1024（text-embedding-v3），非 OpenAI 的 1536
- `SemanticChunker` 中表格/图表/公式为原子块，不参与合并切分
