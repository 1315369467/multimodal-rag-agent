"""
稠密检索器
──────────────────────────────────────────────────────────────────────────────
封装 ChromaVectorStore 的相似度检索，返回 (Document, 归一化分数) 对，
以便与稀疏检索器的结果进行融合。
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()


class DenseRetriever:
    """
    基于 Chroma 向量相似度的检索器。

    返回 (Document, 归一化分数) 对，分数 1.0 表示最相关。
    """

    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """
        按稠密向量相似度检索 top-k 文档。

        参数
        ----
        query  : 原始查询字符串。
        k      : 返回候选数量（默认使用 settings.dense_top_k）。
        filter : 可选的 Chroma 元数据过滤条件。

        返回
        ----
        按分数降序排列的 (Document, 分数) 列表。
        分数为归一化余弦相似度，范围 ∈ [0, 1]。
        """
        k = k or settings.dense_top_k
        raw: list[tuple[Document, float]] = self._store.similarity_search_with_score(
            query, k=k, filter=filter
        )

        # Chroma 返回 L2 距离；转换为余弦相似度风格的分数
        # score = 1 / (1 + distance)，距离越小分数越高
        scored = [
            (doc, 1.0 / (1.0 + max(dist, 0.0)))
            for doc, dist in raw
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"[DenseRetriever] query='{query[:60]}' → {len(scored)} 条结果 "
            f"(最高分: {scored[0][1]:.3f})" if scored else
            f"[DenseRetriever] query='{query[:60]}' → 0 条结果"
        )
        return scored
