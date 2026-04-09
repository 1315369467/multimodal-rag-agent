# 多模态 RAG 本地知识库问答 Agent

企业级多模态文档解析 → RAG 问答全链路系统。支持 PDF、TXT、Markdown、扫描件、图表、公式等复杂排版文档的结构化检索与智能问答。内置 Streamlit 可视化前端，支持知识库浏览、增删改查及交互式问答。


## 架构总览

```
输入文档 (PDF / TXT / Markdown / 图片)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              DocumentRouter  (路由层)                │
│                                                     │
│  文本页面 → PDFTextParser (PyMuPDF + pdfplumber)    │
│  纯文本   → TextParser    (TXT / Markdown 解析)     │
│  图像页面 → VisionParser  (Qwen-VL 多模态理解)      │
└──────────────────────┬──────────────────────────────┘
                       │ ParsedBlock[]
                       ▼
┌─────────────────────────────────────────────────────┐
│           SemanticChunker  (语义感知切分)            │
│  • 表格/图注/公式 → 原子 chunk (不切断)              │
│  • 标题作为 context 附加到后续 chunk                 │
│  • 基于版面坐标 + 字体层级决定切分边界               │
└──────────────────────┬──────────────────────────────┘
                       │ LangChain Document[]
                       ▼
┌─────────────────────────────────────────────────────┐
│           ChromaVectorStore  (持久化向量库)          │
│  本地模式：Qwen3-Embedding-0.6B → 1024维向量        │
│  API 模式：text-embedding-v3   → 1024维向量         │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
  DenseRetriever             SparseRetriever
  (Chroma 向量检索)          (BM25 词汇检索)
           │                        │
           └───────────┬────────────┘
                       ▼
              RRF Fusion (倒数排名融合)
                       │
                       ▼
          （可选）QwenReranker 精排
                       │
                       ▼
          MultimodalRAGAgent (LangGraph ReAct)
          Qwen3-235B 生成最终答案
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI 后端 (REST + SSE)  ←→  Streamlit 前端      │
└─────────────────────────────────────────────────────┘
```

## 模型说明

| 角色 | 模型 | 说明 |
|------|------|------|
| 主 LLM | `qwen3-235b-a22b` | 问答生成、推理 |
| 视觉理解 | `qwen-vl-max` | 扫描页/图表/公式转录 |
| 向量化（本地，默认） | `Qwen/Qwen3-Embedding-0.6B` | 本地推理，无需 API（1024 维） |
| 向量化（API 模式） | `text-embedding-v3` | DashScope API 调用（1024 维） |
| 重排序（可选，默认关闭） | `gte-rerank` / `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder 精排，可按模式切换 |

主 LLM 和视觉模型通过 DashScope OpenAI-compatible API 调用；Embedding 和 Reranker 支持本地/API 两种模式，可在配置文件中切换。

## Agent 工具清单

MultimodalRAGAgent 基于 LangGraph ReAct 框架，配备以下工具：

### 检索工具

| 工具 | 说明 |
|------|------|
| `knowledge_base_search` | 通用知识库混合检索（稠密 + 稀疏 + RRF 融合） |
| `knowledge_base_search_with_filter` | 限定来源文件的检索（精准定位特定文档） |
| `query_rewrite` | 将复杂问题改写为多个子查询，提升命中率 |
| `multi_round_search` | 对子查询列表依次检索并合并去重结果 |

### 扩展技能（Skills）

| 工具 | 说明 |
|------|------|
| `calculator` | 安全数学表达式求值（不依赖 LLM 口算） |
| `web_search` | DuckDuckGo 实时网络搜索（知识库无结果时补充） |
| `table_analyzer` | Markdown/CSV 表格统计分析（行列汇总、均值等） |
| `image_describer` | Qwen-VL 图像内容描述（理解检索结果中的图像路径） |
| `summarizer` | LLM 文本摘要压缩（检索片段过长时先压缩再组合） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

主要配置项（`config/settings.py` 或 `.env` 均可覆盖）：

```env
DASHSCOPE_API_KEY=sk-xxx          # 必填

# Embedding 模式（local=本地推理，api=DashScope API）
EMBEDDING_MODE=local              # 默认 local
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# 是否启用 Reranker 精排（False=跳过，节省延迟）
ENABLE_RERANK=false               # 默认关闭

# Chunk 切分策略
CHUNK_STRATEGY=semantic           # fixed|sentence|paragraph|markdown|recursive|semantic
```

### 3. 文档入库

```bash
# 批量入库目录下所有文档
python scripts/ingest_documents.py --input-dir ./data/documents

# 指定文件入库（支持 PDF / TXT / MD / PNG / JPG）
python scripts/ingest_documents.py --files report.pdf notes.md data.txt

# 清空后重新入库
python scripts/ingest_documents.py --input-dir ./data/documents --reset
```

### 4. 启动 API 服务

```bash
# 开发模式（热重载）
python run_server.py --reload

# 生产模式
python run_server.py --host 0.0.0.0 --port 8000 --workers 4
```

### 5. 启动前端

```bash
streamlit run frontend/app.py
```

前端提供四个功能模块：

| 模块 | 功能 |
|------|------|
| 知识库浏览 | 分页列表、展开查看、单条/批量删除 |
| 智能问答 | 流式 Agent 问答（实时显示工具调用步骤）、来源引用、多轮对话 |
| 文档入库 | 上传文件（PDF/TXT/MD/PNG/JPG）、查看处理结果 |
| 系统管理 | 健康状态、文档统计、重置集合 |

### 6. API 接口

```bash
# 健康检查
curl http://localhost:8000/health

# 问答（同步）
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "公司2023年的营业收入是多少？"}'

# 流式问答（SSE，实时推送工具调用过程）
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "公司主营业务是什么？"}'

# 上传并入库文件
curl -X POST http://localhost:8000/v1/ingest/upload \
  -F "files=@report.pdf" \
  -F "files=@notes.md"

# 多轮对话
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请详细解释一下",
    "chat_history": [
      {"role": "user", "content": "公司主营业务是什么？"},
      {"role": "assistant", "content": "公司主营..."}
    ]
  }'

# 分页获取文档列表
curl "http://localhost:8000/v1/documents?offset=0&limit=20"

# 获取单个文档
curl http://localhost:8000/v1/documents/{doc_id}

# 更新文档
curl -X PUT http://localhost:8000/v1/documents/{doc_id} \
  -H "Content-Type: application/json" \
  -d '{"content": "更新后的内容", "metadata": {"source": "doc.pdf"}}'

# 删除文档
curl -X DELETE http://localhost:8000/v1/documents/{doc_id}

# 重置集合
curl -X DELETE http://localhost:8000/v1/collection
```

#### 流式接口事件格式

`POST /v1/query/stream` 以 Server-Sent Events 格式逐步推送 Agent 思考过程：

```
data: {"type": "tool_start", "step_idx": 0, "tool": "knowledge_base_search", "args": {...}}
data: {"type": "tool_done",  "step_idx": 0, "tool": "knowledge_base_search", "elapsed_ms": 312, "result_summary": "找到 3 个片段…"}
data: {"type": "answer", "answer": "...", "sources": [...], "total_elapsed_ms": 1500}
data: [DONE]
```

### 7. 运行评估

```bash
# 离线模式（BM25，无需 API Key）
python scripts/evaluate.py \
    --dataset ./data/eval_dataset.jsonl \
    --corpus  ./data/eval_corpus \
    --k 1 3 5 10 \
    --output  ./results/eval_results.json

# 在线模式（完整 Dense + Sparse + Reranker 流水线）
python scripts/evaluate.py --online --dataset ./data/eval_dataset.jsonl

# 检索消融：仅 Dense / 仅 Sparse
python scripts/evaluate.py --online --ablation dense
python scripts/evaluate.py --online --ablation sparse
```

评估数据集格式（JSONL，每行一条）：
```jsonl
{"question": "什么是RAG？", "answer": "检索增强生成...", "source": "intro.pdf", "page": 5}
```

#### Chunk 切分方法消融实验

评估不同文档切分策略对检索性能的影响：

```bash
# 单一策略评估
python scripts/evaluate.py --chunk-ablation fixed      # 固定字符数切分
python scripts/evaluate.py --chunk-ablation sentence   # 句子边界切分
python scripts/evaluate.py --chunk-ablation paragraph  # 段落切分
python scripts/evaluate.py --chunk-ablation markdown   # Markdown 标题切分
python scripts/evaluate.py --chunk-ablation recursive  # 递归字符切分

# 全部策略对比，输出汇总表
python scripts/evaluate.py --chunk-ablation all

# 指定 chunk 大小（token 近似值）
python scripts/evaluate.py --chunk-ablation all --chunk-size 512
```

汇总表示例：

```
  ── K = 5 ──
  策略                  Chunks    Recall       MRR  Precision      NDCG     耗时
  ──────────────────────────────────────────────────────────────────────────────
  Fixed-size              1280    62.50%    0.5412     12.50%    0.6104    3.20s
  Sentence                 940    70.00%    0.6100     14.00%    0.6850    2.80s
  Paragraph                320    75.00%    0.6583     15.00%    0.7200    2.10s
  Markdown-header          180    85.00%    0.7800     17.00%    0.8300    1.50s
  Recursive                260    80.00%    0.7200     16.00%    0.7900    1.90s

  最优策略（按 Recall@5）：Markdown-header (85.00%)
```

消融结果自动保存至 `eval_results_chunk_ablation.json`。

### 8. 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
multimodal-rag-agent/
├── config/
│   └── settings.py              # pydantic-settings 全局配置
├── src/
│   ├── document_parser/
│   │   ├── base_parser.py       # ParsedBlock 数据结构 + 抽象基类
│   │   ├── pdf_parser.py        # PyMuPDF + pdfplumber 文本提取
│   │   ├── text_parser.py       # TXT / Markdown 纯文本解析
│   │   ├── vision_parser.py     # Qwen-VL 视觉理解
│   │   ├── router.py            # 多通道路由编排
│   │   └── chunker.py           # 版面感知语义切分
│   ├── embeddings/
│   │   └── qwen_embedding.py    # Qwen3-Embedding LangChain 封装（本地/API 双模式）
│   ├── retrieval/
│   │   ├── dense_retriever.py   # 稠密向量检索
│   │   ├── sparse_retriever.py  # BM25 稀疏检索
│   │   ├── hybrid_retriever.py  # RRF 融合 + Reranker 编排
│   │   └── reranker.py          # Qwen3-Reranker（本地/API 双模式，可选）
│   ├── vectorstore/
│   │   └── chroma_store.py      # Chroma 持久化向量库 + CRUD
│   ├── agent/
│   │   ├── rag_agent.py         # LangGraph ReAct Agent（同步 + 流式）
│   │   ├── tools.py             # 检索工具（knowledge_base_search 等）
│   │   └── skills/
│   │       ├── __init__.py      # build_skill_tools 工厂函数
│   │       ├── calculator.py    # 安全数学表达式求值
│   │       ├── web_search.py    # DuckDuckGo 实时搜索
│   │       ├── table_analyzer.py# Markdown/CSV 表格统计分析
│   │       ├── image_describer.py# Qwen-VL 图像内容描述
│   │       └── summarizer.py    # LLM 文本摘要压缩
│   └── api/
│       ├── main.py              # FastAPI 应用（REST + SSE 流式接口）
│       └── schemas.py           # Pydantic 请求/响应模型
├── frontend/
│   ├── app.py                   # Streamlit 可视化前端
│   └── style.css                # 自定义样式
├── scripts/
│   ├── ingest_documents.py      # 批量文档入库 CLI
│   └── evaluate.py              # Recall/MRR/Precision 评估
├── tests/
│   ├── test_parser.py           # 解析层单元测试（含 TextParser）
│   ├── test_retrieval.py        # 检索层单元测试
│   └── test_agent.py            # Agent 层单元测试
├── docs/
│   └── qwen_api_models.md       # Qwen API 模型参考文档
├── run_server.py                # uvicorn 启动入口
├── requirements.txt
└── .env.example
```

## 关键设计决策

### 多通道解析
PDF 页面首先经过文本提取（低延迟），若检测到图像覆盖率超过阈值或文字量不足，自动切换到 Qwen-VL 视觉通道，确保图表/扫描页内容不丢失。同时支持 TXT/Markdown 文件的直接解析，Markdown 标题自动识别为 HEADER 块。

### Embedding 双模式
系统支持本地推理和 API 调用两种 Embedding 模式：
- **local（默认）**：加载 `Qwen/Qwen3-Embedding-0.6B` 本地模型，完全离线，适合隐私敏感场景
- **api**：调用 DashScope `text-embedding-v3` 接口，无需 GPU，延迟更低

通过环境变量 `EMBEDDING_MODE=local|api` 切换，无需改动代码。

### 可配置 Chunk 切分策略
系统内置六种切分策略，可在 `config/settings.py` 中通过 `chunk_strategy` 字段或环境变量 `CHUNK_STRATEGY` 切换：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `fixed` | 固定字符数切分，无重叠 | 均匀文本，对粒度敏感 |
| `sentence` | 按句子边界，每块最多 5 句 | 短问答、新闻类文档 |
| `paragraph` | 按连续空行，短段自动合并 | 普通文章、报告 |
| `markdown` | 按 `##`/`###` 标题 | Markdown/结构化文档 |
| `recursive` | LangChain 递归切分，优先段落→句子→字符 | 混合格式文档 |
| `semantic` | ParsedBlock 感知切分，保留版面语义（**默认**） | PDF/扫描件/结构复杂文档 |

#### 关于 `chunk_overlap`

只有 `recursive` 和 `semantic` 两种策略实际使用 `chunk_overlap`，其余策略均无重叠：

| 策略 | 是否使用重叠 | 原因 |
|------|------------|------|
| `fixed` | 否 | 切分点是任意字符位置，加重叠只会产生重复内容 |
| `sentence` | 否 | 句子是最小完整语义单元，重叠会破坏完整性 |
| `paragraph` | 否 | 段落是作者有意划定的语义边界，跨段落重叠引入主题噪音 |
| `markdown` | 否 | 标题已作为 `header_context` 提供上下文锚点 |
| `recursive` | **是** | 切断点可能落在句子中间，重叠保证跨块上下文 |
| `semantic` | **是** | 长文本块边界不一定是完整语义断点，重叠弥补跨块语义断层 |

### 语义感知切分（semantic 策略）

`semantic` 策略由 `SemanticChunker` 实现，以 `ParsedBlock` 列表为输入，逐块处理并维护一个滚动缓冲区：

```
ParsedBlock[]
     │
     ▼ 按 block_type 分三类处理
     │
     ├─ HEADER 块
     │    刷出缓冲区 → 重置缓冲区
     │    将标题文本保存为 header_context（附加到后续所有 chunk）
     │
     ├─ 原子块（TABLE / FIGURE / FORMULA）
     │    刷出缓冲区 → 重置缓冲区
     │    该块单独输出为一个 chunk
     │    header_context 前置拼入内容，保留章节归属
     │
     └─ 普通文本块（TEXT）
          │
          ├─ 跨页？→ 刷出缓冲区，重置
          │
          ├─ 加入当前块后超过 chunk_size？
          │    刷出缓冲区
          │    从上一个 chunk 末尾取 chunk_overlap 个词作为新缓冲区起始
          │
          └─ 追加到缓冲区，记录 block_type
               │
               ▼（遍历结束后刷出剩余缓冲区）
          LangChain Document
          metadata: source / page_num / header_context
                    block_types / chunk_tokens
```

**关键设计点：**

- **标题即上下文**：遇到 HEADER 块时不单独输出，而是将标题文本存入 `header_context`，拼入后续所有 chunk 的开头，使每个 chunk 都能回溯所属章节
- **原子块不切断**：TABLE / FIGURE / FORMULA 整体输出为一个独立 chunk，防止表格行或公式被切断
- **跨页刷新**：页码变化时强制刷出缓冲区，避免将不同页的内容混入同一 chunk
- **重叠保留上下文**：超出 `chunk_size` 时，从上一 chunk 末尾取 `chunk_overlap` 个词作为新 chunk 起始

### RRF 融合
倒数排名融合（RRF）对稠密和稀疏两路检索结果做无参数融合，天然处理两路分数分布不同的问题，比线性加权分数插值更鲁棒。

### 重排精排（可选）
`ENABLE_RERANK=true` 时，Qwen3-Reranker cross-encoder 对 RRF 候选集（默认 top-10）做精排，从 Recall 优先转向 Precision 优先，最终返回 top-5。支持本地模型（`Qwen/Qwen3-Reranker-0.6B`）和 API（`gte-rerank`）两种模式。默认关闭以降低延迟。

### 流式 Agent 响应
`POST /v1/query/stream` 接口以 SSE 格式实时推送 Agent 思考步骤（工具调用开始/完成事件），前端可据此渲染动态进度面板，无需等待完整响应。

### 前端可视化
Streamlit 前端通过 REST API 与后端交互，实时消费 SSE 流，直观展示每个工具调用的耗时与结果摘要，同时提供知识库浏览（分页、展开、搜索）、文档增删改查、对话式问答、系统管理等完整功能。
