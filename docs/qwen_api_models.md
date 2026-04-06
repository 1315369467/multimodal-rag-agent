# Qwen 系列模型 API 速查表

> 更新日期：2026-04-06 | 数据来源：阿里云百炼官方文档

---

## 一、文本对话模型（LLM）

| 系列 | API 名称 | 参数量 | 上下文窗口 | 特点 |
|------|----------|--------|-----------|------|
| Qwen3 旗舰 | `qwen3-235b-a22b` | 235B（激活22B） | 128K | MoE架构，性价比最高旗舰 |
| Qwen3 旗舰 | `qwen3-32b` | 32B | 128K | 稠密模型，推理能力强 |
| Qwen3 旗舰 | `qwen3-14b` | 14B | 128K | 中等规模，速度与能力均衡 |
| Qwen3 旗舰 | `qwen3-8b` | 8B | 128K | 轻量，适合低延迟场景 |
| Qwen3 Max | `qwen3-max` | - | 128K | 托管旗舰，自动指向最新快照 |
| Qwen3 Max | `qwen3-max-2026-01-23` | - | 128K | 固定快照版本，生产推荐 |
| Qwen3.5 Plus | `qwen3.6-plus` | - | 128K | 支持文本+图像+视频 |
| Qwen3.5 Plus | `qwen3.5-plus-2026-02-15` | - | 128K | 固定快照 |
| Qwen Plus | `qwen-plus` / `qwen-plus-latest` | - | 128K | 通用高质量，成本适中 |
| Qwen Turbo | `qwen-turbo` / `qwen-turbo-latest` | - | 128K | 速度快，成本低 |
| Qwen Flash | `qwen3.5-flash` / `qwen-flash` | - | 32K | 超低延迟，适合高并发 |

---

## 二、视觉多模态模型（VL）

| 系列 | API 名称 | 输入模态 | 上下文窗口 | 特点 |
|------|----------|---------|-----------|------|
| Qwen3-VL Plus | `qwen3-vl-plus` | 图像/视频/文本 | 128K | 当前最强视觉模型 |
| Qwen3-VL Flash | `qwen3-vl-flash` | 图像/视频/文本 | 32K | 快速视觉理解 |
| Qwen3-VL Flash | `qwen3-vl-flash-2026-01-22` | 图像/视频/文本 | 32K | 固定快照，支持上下文缓存 |
| Qwen2.5-VL Max | `qwen-vl-max` | 图像/视频/文本 | 32K | 上一代旗舰，仍可用 |
| Qwen2.5-VL Max | `qwen-vl-max-2025-01-25` | 图像/视频/文本 | 32K | 固定快照 |
| Qwen2.5-VL Plus | `qwen-vl-plus` / `qwen-vl-plus-2025-01-25` | 图像/文本 | 32K | 上一代均衡版 |

---

## 三、代码模型（Coder）

| API 名称 | 特点 |
|----------|------|
| `qwen3-coder-plus` | 代码生成旗舰，支持长上下文代码补全 |
| `qwen3-coder-plus-2025-07-22` | 固定快照 |
| `qwen3-coder-flash` | 代码快速补全，低延迟 |
| `qwen3-coder-flash-2025-07-28` | 固定快照 |

---

## 四、Embedding 向量模型

| API 名称 | 向量维度 | 最大输入 Token | 特点 |
|----------|---------|--------------|------|
| `text-embedding-v4` | 2048 | 8192 | **最新最强**，支持 Matryoshka 截断 |
| `text-embedding-v3` | 1024 / 1536 / 3072 | 8192 | 主流版本，多维度可选 |
| `text-embedding-v2` | 1536 | 2048 | 旧版，兼容存量项目 |
| `text-embedding-v1` | 1536 | 2048 | 已不推荐 |

> **Matryoshka（套娃）维度**：`text-embedding-v4` 支持在不重新计算的情况下截断至低维，适合存储受限场景。

---

## 五、Reranker 重排序模型

| API 名称 | 特点 |
|----------|------|
| `gte-rerank` / `gte-rerank-v2` | **当前推荐**，Cross-encoder 精排，多语言 |
| `gte-rerank-v1` | 旧版，已不推荐 |

---

## 六、模型横向对比

### 6.1 文本 LLM 对比

| 维度 | `qwen3-235b-a22b` | `qwen3-max` | `qwen-plus` | `qwen-turbo` | `qwen3.5-flash` |
|------|--------------------|-------------|-------------|-------------|-----------------|
| 定位 | 自托管旗舰 | 托管旗舰 | 均衡通用 | 快速经济 | 超低延迟 |
| 推理能力 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 生成速度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| API 成本 | 高 | 高 | 中 | 低 | 极低 |
| 适用场景 | 复杂推理/文档分析 | 复杂推理 | RAG问答/摘要 | 分类/抽取 | 实时对话/路由 |
| **RAG推荐** | ✅ 首选 | ✅ 备选 | ✅ 经济方案 | ⚠️ 简单场景 | ❌ 不推荐 |

### 6.2 视觉 VL 模型对比

| 维度 | `qwen3-vl-plus` | `qwen3-vl-flash` | `qwen-vl-max`（旧） |
|------|-----------------|------------------|----------------------|
| 图像理解 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 视频理解 | ✅ | ✅ | ✅ |
| OCR/表格 | 最强 | 良好 | 良好 |
| 数学公式 | 最强 | 较好 | 较好 |
| 响应速度 | 中 | 快 | 中 |
| **RAG推荐** | ✅ 扫描/图表解析 | ✅ 批量预处理 | ⚠️ 已有历史数据时兼容 |

### 6.3 Embedding 模型对比

| 维度 | `text-embedding-v4` | `text-embedding-v3` | `text-embedding-v2` |
|------|---------------------|---------------------|---------------------|
| 检索质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 最大维度 | 2048 | 3072 | 1536 |
| 可变维度 | ✅（Matryoshka） | ✅ | ❌ |
| 最大 Token | 8192 | 8192 | 2048 |
| **RAG推荐** | ✅ 新项目首选 | ✅ 现有项目可用 | ❌ 不推荐新用 |

---

## 七、本项目当前配置 vs 推荐配置

| 配置项 | 当前值（`.env`） | 推荐升级 | 说明 |
|--------|----------------|----------|------|
| `LLM_MODEL` | `qwen3-235b-a22b` | `qwen3-max` | 别名更稳定，自动跟随最新版 |
| `VL_MODEL` | `qwen-vl-max` | `qwen3-vl-plus` | 新一代VL模型，图表/公式解析更强 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | `text-embedding-v4` | 新模型检索质量更高，维度更灵活 |
| `RERANKER_MODEL` | `gte-rerank` | ✅ 无需修改 | 已是当前最新 |

> 升级只需修改 `.env` 文件对应的值，代码无需改动。

---

## 八、调用入口

所有模型统一通过 DashScope OpenAI 兼容接口调用：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

| 功能 | Endpoint |
|------|----------|
| Chat / VL | `POST /v1/chat/completions` |
| Embedding | `POST /v1/embeddings` |
| Rerank | `POST /v1/rerank` |

---

*参考来源：[阿里云百炼模型大全](https://help.aliyun.com/zh/model-studio/models) · [千问API参考](https://help.aliyun.com/zh/model-studio/qwen-api-reference/) · [Qwen官方博客](https://qwenlm.github.io/blog/qwen3/)*
