# 多模态 Agentic RAG 知识库问答系统

企业级多模态文档解析 → RAG 问答全链路系统。复杂排版文档（含表格、图表、公式、扫描件）经 MinerU OCR 预处理统一转为结构化 Markdown，再通过语义感知切分写入文本向量库；文档中引用的图片另走 Qwen3-VL-Embedding 跨模态向量化，存入独立图像集合，支持以文搜图。最终由 LangGraph ReAct Agent 统一编排文本与图像两条检索路径。内置 Streamlit 可视化前端，支持知识库浏览、图片检索、增删改查及交互式问答。


## 架构总览

```
输入文档 (PDF / 扫描件 / 图表)
       │
       ▼ MinerU OCR 预处理（布局识别 + OCR）
       │
结构化 Markdown (.md)  +  提取出的图片 (images/*.png)
       │                              │
       ▼                              ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  TextParser             │  │  ingest_images_from_md.py    │
│  Markdown 结构化解析     │  │  扫描 md 中 ![](path) 引用,   │
│  → ParsedBlock[]        │  │  取出本地图片去重              │
│  HEADER / TEXT / TABLE  │  └───────────────┬──────────────┘
│  / FIGURE               │                  │
└───────────┬─────────────┘                  │
            ▼                                ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  StructureAwareChunker  │  │  QwenVLLocalEmbeddings       │
│  结构感知切分            │  │  Qwen3-VL-Embedding-2B       │
│  • 表格/图注原子块       │  │  图片 / 文本 → 同一向量空间    │
│  • 标题作 context        │ └───────────────┬──────────────┘
└───────────┬─────────────┘                  │
            ▼                                ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  ChromaVectorStore      │  │  ImageChromaStore            │
│  文本集合                │  │  图片集合（独立）              │
│  Qwen3-Embedding-0.6B   │  │  VL 跨模态向量                │
│  → 1024 维              │  └───────────────┬──────────────┘
└───────────┬─────────────┘                  │
            │                                │
   ┌────────┴─────────┐                      │
   ▼                  ▼                      │
DenseRetriever   SparseRetriever             │
(Chroma 向量)    (BM25 词汇)                  │
   └────────┬─────────┘                      │
            ▼                                │
     RRF 融合 → (可选) QwenReranker 精排      │
            │                                │
            ▼                                ▼
┌─────────────────────────────────────────────────────┐
│       HybridRetriever.retrieve / retrieve_images    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
         MultimodalRAGAgent (LangGraph ReAct)
         工具：knowledge_base_search / image_search /
               query_rewrite / multi_round_search / skills
         LLM：Qwen3（DashScope 或本地推理）
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI 后端 (REST + SSE)  ←→  Streamlit 前端       │
└─────────────────────────────────────────────────────┘
```

## 模型说明

| 角色 | 模型 | 说明 |
|------|------|------|
| 主 LLM（API，默认） | `qwen3.6-plus` | 问答生成、推理（DashScope API） |
| 主 LLM（本地模式） | `qwen3.5-9b` | 本地部署，通过 `LLM_MODE=local` 切换 |
| 视觉理解（图片→文本） | `qwen3-vl-plus` | VisionParser / image_describer skill 调用 |
| 文本 Embedding（本地，默认） | `Qwen/Qwen3-Embedding-0.6B` | 本地推理，1024 维 |
| 文本 Embedding（API 模式） | `text-embedding-v4` | DashScope API 调用（1024 维） |
| **图像/跨模态 Embedding** | **`Qwen/Qwen3-VL-Embedding-2B`** | **本地推理，图像与文本共享同一向量空间** |
| 重排序（本地，默认） | `Qwen/Qwen3-Reranker-0.6B` | 本地 cross-encoder 精排 |
| 重排序（API 模式） | `qwen3-rerank` | DashScope API 调用，通过 `RERANKER_MODE=api` 切换 |

主 LLM 与视觉理解模型通过 DashScope OpenAI-compatible API 调用；文本 Embedding 和 Reranker 支持本地/API 两种模式；**图像 Embedding 固定本地推理**（Qwen3-VL-Embedding-2B，通过 `trust_remote_code` 加载官方 `Qwen3VLForEmbedding` 脚本）。LLM 同样支持本地部署模式（`LLM_MODE=local`），通过 `LOCAL_LLM_BASE_URL` 指向本地推理服务。

## Agent 工具清单

MultimodalRAGAgent 基于 LangGraph ReAct 框架，配备以下工具：

### 检索工具

| 工具 | 说明 |
|------|------|
| `knowledge_base_search` | 文本知识库混合检索（稠密 + 稀疏 + RRF 融合 + 可选精排） |
| `knowledge_base_search_with_filter` | 限定来源文件的文本检索（精准定位特定文档） |
| `image_search` | **图像库跨模态检索**：文本 query → Qwen3-VL-Embedding → 相关图片 |
| `query_rewrite` | 将复杂问题改写为多个子查询，提升命中率 |
| `multi_round_search` | 对子查询列表依次检索并合并去重结果 |

文本与图像走完全独立的两路检索——`knowledge_base_search` 只返回文本片段（不混入图片），`image_search` 只返回图片。Agent 根据问题语义自行决定调用哪一个或两个都调。

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

# 文本 Embedding 模式（local=本地推理，api=DashScope API）
EMBEDDING_MODE=local              # 默认 local
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# 图像/跨模态 Embedding（固定本地推理）
LOCAL_VL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
CHROMA_IMAGE_COLLECTION_NAME=multimodal_rag_images
IMAGE_TOP_K=5                     # image_search 默认返回数量

# 是否启用 Reranker 精排（False=跳过，节省延迟）
ENABLE_RERANK=true                # 默认开启

# Chunk 切分策略
CHUNK_STRATEGY=semantic           # fixed|sentence|paragraph|markdown|recursive|semantic
```

### 3. 文档预处理（MinerU OCR）

PDF、扫描件等复杂排版文档需先通过 MinerU 转换为结构化 Markdown：

```bash
# 对 data/documents 下所有文档执行 OCR，输出到 data/ocr_output
python scripts/mineru_ocr.py --input-dir ./data/documents
```

输出目录结构：
```
data/ocr_output/
  {文档名}/
    {backend}/
      {文档名}.md      ← 入库目标
      images/          ← 提取出的图片
```

### 4. 文档入库（文本向量库）

```bash
# 批量入库 OCR 输出目录下所有 Markdown
python scripts/ingest_ocr_output.py

# 指定其他 OCR 输出目录
python scripts/ingest_ocr_output.py --ocr-dir ./my_ocr_results

# 清空后重新入库
python scripts/ingest_ocr_output.py --reset

# 指定切分策略
python scripts/ingest_ocr_output.py --chunk-strategy markdown
```

### 5. 图片入库（图像向量库）

将 Markdown 中引用的本地图片通过 Qwen3-VL-Embedding 向量化，存入独立的图片集合，后续即可通过 `image_search` 工具或 `/v1/images/search` 接口以文搜图：

```bash
# 扫描 OCR 输出目录下所有 md 的 ![](path) 引用并入库
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output

# 指定单个 md 文件
python scripts/ingest_images_from_md.py --files ./data/ocr_output/report.md

# 重置图片集合后重新入库
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output --reset
```

特性：
- 仅处理本地相对/绝对路径的图片（http(s):// URL 自动跳过）；
- 同一图片被多个 md 引用时自动去重（以图片绝对路径作为 Chroma ID，幂等写入）；
- 图片的 `alt_text` / `caption` / 来源 md 文件名作为 metadata 一同保存。

可选：清理未被任何 md 引用的孤儿图片：

```bash
python scripts/cleanup_unused_images.py --dry-run   # 先预览
python scripts/cleanup_unused_images.py             # 确认后实际删除
```

### 6. 启动 API 服务

```bash
# 开发模式（热重载）
python run_server.py --reload

# 生产模式
python run_server.py --host 0.0.0.0 --port 8000 --workers 4
```

服务启动时会同时加载 `ChromaVectorStore`（文本）和 `ImageChromaStore`（图片）两个集合，后者会懒加载 Qwen3-VL-Embedding 模型。

### 7. 启动前端

```bash
streamlit run frontend/app.py
```

前端提供五个功能模块：

| 模块 | 功能 |
|------|------|
| 智能问答 | 流式 Agent 问答（实时显示工具调用步骤）、来源引用、多轮对话 |
| 图片检索 | 以文搜图画廊展示，显示相似度分数、alt / caption、来源 md |
| 知识库浏览 | 分页列表、展开查看、单条/批量删除 |
| 文档入库 | 上传文件（TXT/MD/PNG/JPG）、查看处理结果 |
| 系统管理 | 健康状态、文档与图片统计、重置集合 |

### 8. API 接口

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

# 上传并入库文件（支持 TXT / MD / PNG / JPG）
curl -X POST http://localhost:8000/v1/ingest/upload \
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
  -d '{"content": "更新后的内容", "metadata": {"source": "doc.md"}}'

# 删除文档
curl -X DELETE http://localhost:8000/v1/documents/{doc_id}

# 重置文本集合
curl -X DELETE http://localhost:8000/v1/collection

# ── 图片检索接口 ────────────────────────────────────────────

# 跨模态图片检索（文本 query → 相关图片）
curl -X POST http://localhost:8000/v1/images/search \
  -H "Content-Type: application/json" \
  -d '{"query": "模型架构图", "top_k": 5}'

# 获取图片原文件（供前端展示）
curl "http://localhost:8000/v1/images/file?path=/absolute/path/to/image.png" --output img.png

# 查询图片库总数
curl http://localhost:8000/v1/images/count
```

#### 流式接口事件格式

`POST /v1/query/stream` 以 Server-Sent Events 格式逐步推送 Agent 思考过程：

```
data: {"type": "tool_start", "step_idx": 0, "tool": "knowledge_base_search", "args": {...}}
data: {"type": "tool_done",  "step_idx": 0, "tool": "knowledge_base_search", "elapsed_ms": 312, "result_summary": "找到 3 个片段…"}
data: {"type": "answer", "answer": "...", "sources": [...], "total_elapsed_ms": 1500}
data: [DONE]
```

### 9. 运行评估

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
{"question": "什么是RAG？", "answer": "检索增强生成...", "source": "intro.md", "page": 5}
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

### 10. 运行测试

```bash
pytest tests/ -v

# 仅诊断 VL embedding 环境（模型加载 / 文本-图像相似度对齐）
python tests/diagnose_vl_embedding.py
```

## 项目结构

```
multimodal-rag-agent/
├── config/
│   └── settings.py              # pydantic-settings 全局配置
├── src/
│   ├── document_parser/
│   │   ├── base_parser.py       # ParsedBlock 数据结构 + 抽象基类
│   │   ├── text_parser.py       # Markdown / TXT 结构化解析
│   │   ├── vision_parser.py     # Qwen-VL 视觉理解（独立图像文件转 md）
│   │   ├── router.py            # 文件类型路由
│   │   └── chunker.py           # 版面感知语义切分
│   ├── embeddings/
│   │   ├── qwen_embedding.py    # Qwen3-Embedding（文本，本地/API 双模式）
│   │   └── qwen_vl_embedding.py # Qwen3-VL-Embedding（图像 + 文本跨模态）
│   ├── retrieval/
│   │   ├── dense_retriever.py   # 稠密向量检索
│   │   ├── sparse_retriever.py  # BM25 稀疏检索
│   │   ├── hybrid_retriever.py  # RRF 融合 + Reranker + 图像检索编排
│   │   └── reranker.py          # Qwen3-Reranker（本地/API 双模式，可选）
│   ├── vectorstore/
│   │   ├── chroma_store.py      # 文本向量库 + CRUD
│   │   └── image_store.py       # 图片向量库（跨模态 Chroma 集合）
│   ├── agent/
│   │   ├── rag_agent.py         # LangGraph ReAct Agent（同步 + 流式）
│   │   ├── tools.py             # 检索工具（knowledge_base_search / image_search 等）
│   │   └── skills/
│   │       ├── __init__.py      # build_skill_tools 工厂函数
│   │       ├── calculator.py    # 安全数学表达式求值
│   │       ├── web_search.py    # DuckDuckGo 实时搜索
│   │       ├── table_analyzer.py# Markdown/CSV 表格统计分析
│   │       ├── image_describer.py# Qwen-VL 图像内容描述
│   │       └── summarizer.py    # LLM 文本摘要压缩
│   └── api/
│       ├── main.py              # FastAPI 应用（REST + SSE + 图片接口）
│       └── schemas.py           # Pydantic 请求/响应模型
├── frontend/
│   ├── app.py                   # Streamlit 前端（问答 / 图片检索 / 管理）
│   └── style.css                # 自定义样式
├── scripts/
│   ├── mineru_ocr.py            # MinerU OCR 预处理（PDF → Markdown）
│   ├── ingest_ocr_output.py     # OCR 结果批量入库 CLI（文本集合）
│   ├── ingest_images_from_md.py # 扫描 md 中图片引用并入图像集合
│   ├── cleanup_unused_images.py # 清理未被任何 md 引用的孤儿图片
│   ├── ingest_documents.py      # 通用入库脚本（走 DocumentRouter）
│   ├── generate_eval_dataset.py # LLM 自动生成 QA 评估集
│   ├── dedup_dataset.py         # 评估集去重
│   └── evaluate.py              # Recall/MRR/Precision 评估
├── tests/
│   ├── test_parser.py           # 解析层单元测试（含 TextParser）
│   ├── test_retrieval.py        # 检索层单元测试
│   ├── test_agent.py            # Agent 层单元测试
│   └── diagnose_vl_embedding.py # VL embedding 环境诊断脚本
├── docs/
│   └── qwen_api_models.md       # Qwen API 模型参考文档
├── run_server.py                # uvicorn 启动入口
├── requirements.txt
└── .env.example
```

## 评估基准

基于 173 条问答对的在线评估（混合检索，`chunk_size=1024`，`semantic` 策略）：

**混合检索（Dense + Sparse + RRF，无精排）**

| K | Recall | MRR | Precision | NDCG |
|---|--------|-----|-----------|------|
| 1 | 76.30% | 0.7630 | 76.30% | 0.7630 |
| 3 | 89.02% | 0.8227 | 71.68% | 0.8402 |
| 5 | 94.80% | 0.8357 | 68.09% | 0.8638 |
| 10 | 94.80% | 0.8357 | 34.05% | 0.8638 |

**混合检索 + Qwen3-Reranker 精排**

| K | Recall | MRR | Precision | NDCG |
|---|--------|-----|-----------|------|
| 1 | **86.71%** | **0.8671** | **86.71%** | **0.8671** |
| 3 | 94.80% | 0.9066 | 76.49% | 0.9174 |
| 5 | 95.95% | 0.9094 | 69.94% | 0.9223 |
| 10 | 95.95% | 0.9094 | 34.97% | 0.9223 |

开启精排后 Recall@1 从 76.30% 提升至 86.71%，MRR@1 从 0.7630 提升至 0.8671。

---

## 关键设计决策

### 文档解析策略

复杂排版文档（PDF、扫描件）先通过 **MinerU OCR** 进行布局识别与 OCR，统一输出为结构化 Markdown，再交由 `TextParser` 解析。`TextParser` 能识别以下 Markdown 元素并映射为带类型标注的 `ParsedBlock`：

| Markdown 元素 | BlockType | 切分行为 |
|---|---|---|
| `# 标题` | `HEADER` | 不单独输出，作为后续 chunk 的 `header_context` |
| `\| 表格 \|` | `TABLE` | 原子块，整体输出，不跨行截断 |
| `![alt](url)` 独立行 | `FIGURE` | 原子块，整体输出 |
| ` ``` ``` ` 代码块 | `TEXT` | 整体保留，携带 `subtype=code_block` |
| 其余段落 | `TEXT` | 按 chunk_size 滚动切分 |

独立图像文件（PNG/JPG 等）直接由 `VisionParser` 通过 Qwen-VL 转录为 Markdown 文本。Markdown 中通过 `![]()` 引用的本地图片则走另一条独立的图像向量化路径，见下节。

### 跨模态图像检索（双集合设计）

为了在不牺牲文本检索精度的前提下支持以文搜图，系统维护两个独立的 Chroma 集合：

| 集合 | 内容 | Embedding | 典型调用 |
|------|------|-----------|----------|
| `multimodal_rag` | 文本片段（Markdown 切块） | Qwen3-Embedding-0.6B（1024 维） | `knowledge_base_search` |
| `multimodal_rag_images` | 图片 | Qwen3-VL-Embedding-2B（原生维度） | `image_search` |

**关键选择：**

- **两条独立路径，不做融合**：文本走 dense+sparse+RRF+rerank，图片走 VL 向量直接余弦召回。实验表明文本/图像向量分布差异极大，简单 RRF 融合会让图片结果被文本稀释；交由 Agent 根据问题语义自行决定调用哪一个更稳。
- **以图片绝对路径作 Chroma ID**：`scripts/ingest_images_from_md.py` 重复运行幂等，同一图片被多个 md 引用也只写一次。
- **Metadata 保留 alt_text / caption**：便于前端展示，也能在 `knowledge_base_search` 返回的文本片段中通过 `source_path` 字段回指原图。
- **本地推理固定**：VL embedding 模型 Qwen3-VL-Embedding-2B 通过 `trust_remote_code` 动态加载模型仓库里的 `scripts/qwen3_vl_embedding.py`，无 API 版本；系统会自动扫描 HF 缓存目录，兼容 `refs/main` 与实际 snapshot 不一致的场景。

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
| `semantic` | ParsedBlock 感知切分，保留版面语义（**默认**） | 结构复杂的 Markdown 文档 |

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

### 结构感知切分（semantic 策略）

`semantic` 策略由 `StructureAwareChunker` 实现，以 `ParsedBlock` 列表为输入，逐块处理并维护一个滚动缓冲区：

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
- **重叠保留上下文**：超出 `chunk_size` 时，从上一 chunk 末尾取 `chunk_overlap` 个词作为新 chunk 起始

### RRF 融合
倒数排名融合（RRF）对稠密和稀疏两路检索结果做无参数融合，天然处理两路分数分布不同的问题，比线性加权分数插值更鲁棒。

### 重排精排（可选）
`ENABLE_RERANK=true` 时（默认开启），Qwen3-Reranker cross-encoder 对 RRF 候选集（默认 top-10）做精排，从 Recall 优先转向 Precision 优先，最终返回 top-5。支持本地模型（`Qwen/Qwen3-Reranker-0.6B`，默认）和 API（`qwen3-rerank`）两种模式，通过 `RERANKER_MODE=local|api` 切换。如需降低延迟，可设置 `ENABLE_RERANK=false` 关闭精排。

### 流式 Agent 响应
`POST /v1/query/stream` 接口以 SSE 格式实时推送 Agent 思考步骤（工具调用开始/完成事件），前端可据此渲染动态进度面板，无需等待完整响应。

### 前端可视化
Streamlit 前端通过 REST API 与后端交互，实时消费 SSE 流，直观展示每个工具调用的耗时与结果摘要，同时提供知识库浏览（分页、展开、搜索）、文档增删改查、对话式问答、系统管理等完整功能。
