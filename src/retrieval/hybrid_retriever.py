"""
混合检索器（稠密 + 稀疏 → RRF 融合）
──────────────────────────────────────────────────────────────────────────────
通过倒数排名融合（RRF）将稠密向量检索与 BM25 稀疏检索结合，
然后可选地使用 Qwen3-Reranker 进行精排。

文本索引和图像索引现在通过两个独立方法暴露：
  • retrieve(query)         — 仅文本：稠密 + 稀疏 → RRF → Rerank
  • retrieve_images(query)  — 仅图像：ImageChromaStore 跨模态检索
Agent 工具层将它们暴露为独立工具，由 LLM 决定何时调用哪一个。

RRF 公式
────────
  rrf(d) = Σ  1 / (k + rank_i(d))
          列表 i

其中 k=60 是标准常数，用于抑制排名靠前位置的过度优势。
稠密和稀疏列表在融合前分别乘以各自的权重。

该方案的优势：
  • 对稠密/稀疏分数分布差异具有鲁棒性。
  • 除列表权重外无需调整超参数。
  • 在基准测试中持续优于简单分数插值。
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .reranker import QwenReranker

if TYPE_CHECKING:
    from src.vectorstore.image_store import ImageChromaStore

settings = get_settings()

_RRF_K = 60  # 标准 RRF 常数


class HybridRetriever:
    """
    检索子系统的统一入口，暴露两条独立路径。

    文本检索流水线
    ────────────
    query → DenseRetriever   ┐
                             ├─ RRF 融合 → (可选) Reranker → top-k 文本
    query → SparseRetriever  ┘

    图像检索流水线
    ────────────
    query → ImageStore (跨模态) → top-k 图片

    参数
    ----
    dense_retriever  : 已初始化的 DenseRetriever。
    sparse_retriever : 已初始化的 SparseRetriever（需提前构建索引）。
    reranker         : 可选的 QwenReranker（传 None 则禁用精排）。
    image_store      : 可选的 ImageChromaStore（传入后支持 retrieve_images）。
    dense_weight     : RRF 融合前稠密分数的权重（默认 0.7）。
    sparse_weight    : RRF 融合前稀疏分数的权重（默认 0.3）。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: QwenReranker | None = None,
        image_store: "ImageChromaStore | None" = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> None:
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.image_store = image_store
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight

    # ──────────────────────────────────────────────────────────────────────
    # 文本检索
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        rerank_k: int | None = None,
        final_k: int | None = None,
    ) -> list[Document]:
        """
        文本检索流水线：稠密 + 稀疏 → RRF → (可选) Reranker。

        参数
        ----
        query     : 原始查询字符串。
        dense_k   : 稠密召回数量（默认：settings.dense_top_k）。
        sparse_k  : 稀疏召回数量（默认：settings.sparse_top_k）。
        rerank_k  : 送入重排器的候选数量（默认：settings.rerank_top_k）。
        final_k   : 最终返回数量（默认：settings.final_top_k）。

        返回
        ----
        按相关性降序排列的文本 Document 列表。
        """
        dense_k = dense_k or settings.dense_top_k
        sparse_k = sparse_k or settings.sparse_top_k
        rerank_k = rerank_k or settings.rerank_top_k
        final_k = final_k or settings.final_top_k

        dense_results = self.dense.retrieve(query, k=dense_k)
        sparse_results = (
            self.sparse.retrieve(query, k=sparse_k)
            if self.sparse.is_ready
            else []
        )

        fused = self._rrf_fuse(dense_results, sparse_results)
        candidates = fused[: rerank_k if self.reranker else final_k]

        logger.info(
            f"[HybridRetriever] 稠密={len(dense_results)} "
            f"稀疏={len(sparse_results)} → 融合后={len(fused)} 候选"
        )

        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=final_k)
            logger.info(f"[HybridRetriever] 精排后：{len(candidates)} 条")
        else:
            candidates = candidates[:final_k]

        return candidates

    # ──────────────────────────────────────────────────────────────────────
    # 图像检索
    # ──────────────────────────────────────────────────────────────────────

    def retrieve_images(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        图像检索路径：使用 VL 跨模态嵌入从 ImageChromaStore 中检索图片。

        参数
        ----
        query : 自然语言查询。
        k     : 返回数量（默认：settings.image_top_k）。

        返回
        ----
        (Document, 余弦相似度) 列表；若未配置或为空，返回空列表。
        Document.metadata 包含 block_type="figure"、source_path、file_name 等字段。
        """
        if self.image_store is None or self.image_store.count() == 0:
            return []

        k = k or settings.image_top_k
        try:
            return self.image_store.search_by_text(query, k=k)
        except Exception as exc:
            logger.warning(f"[HybridRetriever] 图片检索失败：{exc}")
            return []

    # ──────────────────────────────────────────────────────────────────────
    # RRF 实现
    # ──────────────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[tuple[Document, float]],
        sparse: list[tuple[Document, float]],
    ) -> list[Document]:
        """对稠密和稀疏排序列表执行倒数排名融合。"""
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        def _process_list(
            ranked: list[tuple[Document, float]], weight: float
        ) -> None:
            for rank, (doc, _score) in enumerate(ranked, start=1):
                key = doc.page_content[:200]
                rrf_scores[key] += weight * (1.0 / (_RRF_K + rank))
                doc_map[key] = doc

        _process_list(dense, self.dense_weight)
        _process_list(sparse, self.sparse_weight)

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]
