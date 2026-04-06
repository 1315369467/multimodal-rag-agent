"""
Qwen3-Embedding（DashScope / OpenAI 兼容接口）
──────────────────────────────────────────────────────────────────────────────
将 DashScope text-embedding-v3 模型封装为 LangChain Embeddings 对象。

特性
────
• 批量编码，可配置批大小（API 上限：25 条/次）。
• 通过 tenacity 对瞬时 HTTP 错误自动重试。
• 可选 L2 归一化，使输出向量直接适配余弦相似度计算。
• 线程安全（__init__ 后无状态）。
"""
from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()

_BATCH_SIZE = 25  # DashScope 单次 API 调用的文本条数上限


class QwenEmbeddings(Embeddings):
    """
    基于 DashScope Qwen3-Embedding 的 LangChain 兼容 Embedding 类。

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
