"""
Qwen3-Embedding（支持本地部署 / DashScope API 两种模式）
──────────────────────────────────────────────────────────────────────────────
提供两种 LangChain Embeddings 实现：
• QwenLocalEmbeddings  — 本地 transformers 推理，默认 Qwen/Qwen3-Embedding-0.6B。
• QwenAPIEmbeddings    — 通过 DashScope / OpenAI 兼容 API 远程调用。

通过工厂函数 ``QwenEmbeddings()`` 根据配置自动选择后端。
"""
from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()

_BATCH_SIZE = 25  # DashScope 单次 API 调用的文本条数上限


# ═══════════════════════════════════════════════════════════════════════════
# 本地模型实现
# ═══════════════════════════════════════════════════════════════════════════

class QwenLocalEmbeddings(Embeddings):
    """
    基于 transformers 本地推理的 Embedding 类。

    参数
    ----
    model_name_or_path : HuggingFace 模型名称或本地路径（默认从 settings 读取）。
    dimensions         : 输出向量维度，-1 表示使用模型原始维度。
    max_length         : tokenizer 最大长度。
    batch_size         : 推理时的批大小。
    instruction        : query 编码时的指令前缀。
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        dimensions: int = 1024,
        max_length: int = 4096,
        batch_size: int = 8,
        instruction: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from transformers.utils import is_flash_attn_2_available

        self.model_name = model_name_or_path or settings.local_embedding_model
        self.dimensions = dimensions
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction or "Given a web search query, retrieve relevant passages that answer the query"

        use_cuda = torch.cuda.is_available()
        if is_flash_attn_2_available() and use_cuda:
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True,
                attn_implementation="flash_attention_2", dtype=torch.float16,
                local_files_only=True,
            )
        else:
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True, dtype=torch.float16,
                local_files_only=True,
            )
        if use_cuda:
            self._model = self._model.cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left",
            local_files_only=True,
        )
        logger.info(f"[Embedding] 本地模型已加载: {self.model_name} (cuda={use_cuda})")

    # ── LangChain 接口 ────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _encode(self, sentences: list[str], is_query: bool) -> list[list[float]]:
        import torch
        import torch.nn.functional as F

        if is_query:
            sentences = [
                f"Instruct: {self.instruction}\nQuery:{s}" for s in sentences
            ]

        all_vectors: list[list[float]] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state
                pooled = self._last_token_pool(hidden, inputs["attention_mask"])
                if self.dimensions != -1:
                    pooled = pooled[:, : self.dimensions]
                pooled = F.normalize(pooled, p=2, dim=1)
            all_vectors.extend(pooled.cpu().tolist())
            torch.cuda.empty_cache()

        return all_vectors

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            __import__("torch").arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


# ═══════════════════════════════════════════════════════════════════════════
# API 模型实现
# ═══════════════════════════════════════════════════════════════════════════

class QwenAPIEmbeddings(Embeddings):
    """
    基于 DashScope Qwen3-Embedding 的 LangChain 兼容 Embedding 类（API 模式）。

    参数
    ----
    model      : DashScope 模型名称（默认：text-embedding-v3）。
    normalize  : 是否对输出向量进行 L2 归一化（余弦相似度场景推荐开启）。
    batch_size : 每次 API 调用的文本数量（DashScope 上限 25）。
    dimensions : 传给 API 的向量维度提示（默认 1024）。
    """

    def __init__(
        self,
        model: str | None = None,
        normalize: bool = True,
        batch_size: int = _BATCH_SIZE,
        dimensions: int = 1024,
    ) -> None:
        from openai import OpenAI

        self.model = model or settings.embedding_model
        self.normalize = normalize
        self.batch_size = min(batch_size, _BATCH_SIZE)
        self.dimensions = dimensions
        self._client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    # ──────────────────────────────────────────────────────────────────────
    # LangChain Embeddings 接口
    # ──────────────────────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档字符串列表进行批量向量化，返回浮点向量列表。"""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_batch(batch)
            all_embeddings.extend(vecs)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """对单条查询字符串进行向量化。"""
        return self._embed_batch([text])[0]

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用 DashScope Embedding API 处理一个批次。"""
        logger.debug(f"[Embedding] 使用 {self.model} 编码 {len(texts)} 条文本")

        resp = self._client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )

        # 按 index 排序，确保顺序与输入一致
        vectors = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

        if self.normalize:
            vectors = [self._l2_normalize(v) for v in vectors]

        return vectors

    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        """对向量进行 L2 归一化。"""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vector
        return (arr / norm).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# 工厂函数 — 根据 settings.embedding_mode 自动选择后端
# ═══════════════════════════════════════════════════════════════════════════

def QwenEmbeddings(mode: str | None = None, **kwargs) -> Embeddings:
    """
    根据配置返回对应的 Embedding 实例。

    参数
    ----
    mode : "local" 或 "api"，为 None 时读取 settings.embedding_mode。
    **kwargs : 透传给具体实现类的构造参数。
    """
    mode = (mode or settings.embedding_mode).lower()
    if mode == "local":
        logger.info("[Embedding] 使用本地模型模式")
        return QwenLocalEmbeddings(**kwargs)
    elif mode == "api":
        logger.info("[Embedding] 使用 API 模式")
        return QwenAPIEmbeddings(**kwargs)
    else:
        raise ValueError(f"不支持的 embedding_mode: {mode!r}，请使用 'local' 或 'api'")


#   python -m src.embeddings.qwen_embedding  (从项目根目录运行)
if __name__ == "__main__":
    import sys, time
    # from pathlib import Path
    # sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    model_path = "Qwen/Qwen3-Embedding-0.6B"
    dim = 1024

    queries = ["What is the capital of China?", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a subset of artificial intelligence.",
        "The Great Wall of China is one of the greatest wonders of the world.",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a subset of artificial intelligence.",
        "The Great Wall of China is one of the greatest wonders of the world.",
    ]

    # ── 加载模型 ──────────────────────────────────────────────────────────
    print(f"正在加载模型: {model_path}")
    t0 = time.perf_counter()
    model = QwenLocalEmbeddings(model_name_or_path=model_path, dimensions=dim)
    load_time = time.perf_counter() - t0
    print(f"模型加载耗时: {load_time:.2f}s\n")

    # ── Query 编码 ────────────────────────────────────────────────────────
    print(f"编码 {len(queries)} 条 query ...")
    t0 = time.perf_counter()
    query_vecs = model.embed_documents(queries)  # 走 is_query=False 的通用路径
    query_time = time.perf_counter() - t0
    print(f"  耗时: {query_time * 1000:.1f}ms  ({query_time / len(queries) * 1000:.1f}ms/条)")

    # 也测试 embed_query（带 instruction 前缀）
    t0 = time.perf_counter()
    query_vecs_with_inst = [model.embed_query(q) for q in queries]
    query_inst_time = time.perf_counter() - t0
    print(f"  带 instruction 编码: {query_inst_time * 1000:.1f}ms  ({query_inst_time / len(queries) * 1000:.1f}ms/条)")

    # ── Document 编码 ─────────────────────────────────────────────────────
    print(f"\n编码 {len(documents)} 条 document ...")
    t0 = time.perf_counter()
    doc_vecs = model.embed_documents(documents)
    doc_time = time.perf_counter() - t0
    print(f"  耗时: {doc_time * 1000:.1f}ms  ({doc_time / len(documents) * 1000:.1f}ms/条)")

    # ── 向量维度验证 ──────────────────────────────────────────────────────
    print(f"\n向量维度: {len(doc_vecs[0])} (期望 {dim})")

    # ── 相似度矩阵 ───────────────────────────────────────────────────────
    q_arr = np.array(query_vecs_with_inst)
    d_arr = np.array(doc_vecs)
    scores = (q_arr @ d_arr.T) * 100
    print("\n相似度矩阵 (query x document, ×100):")
    header = "".ljust(30) + "".join(f"doc{j:<8d}" for j in range(len(documents)))
    print(header)
    for i, q in enumerate(queries):
        row = q[:28].ljust(30) + "".join(f"{scores[i][j]:<12.2f}" for j in range(len(documents)))
        print(row)

    # ── 吞吐量估算（较大批量）────────────────────────────────────────────
    large_batch = documents * 20  # 100 条
    print(f"\n吞吐量测试: {len(large_batch)} 条文本 ...")
    t0 = time.perf_counter()
    _ = model.embed_documents(large_batch)
    throughput_time = time.perf_counter() - t0
    throughput = len(large_batch) / throughput_time
    print(f"  耗时: {throughput_time:.2f}s  |  吞吐量: {throughput:.0f} 条/s")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"  模型加载      : {load_time:.2f}s")
    print(f"  Query 编码    : {query_time / len(queries) * 1000:.1f}ms/条")
    print(f"  Document 编码 : {doc_time / len(documents) * 1000:.1f}ms/条")
    print(f"  批量吞吐量    : {throughput:.0f} 条/s ({len(large_batch)} 条)")
    print(f"  向量维度      : {len(doc_vecs[0])}")
