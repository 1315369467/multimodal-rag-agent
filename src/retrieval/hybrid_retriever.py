"""
混合检索器（稠密 + 稀疏 → RRF 融合）
──────────────────────────────────────────────────────────────────────────────
通过倒数排名融合（RRF）将稠密向量检索与 BM25 稀疏检索结合，
然后可选地使用 Qwen3-Reranker 进行精排。

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

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .reranker import QwenReranker

settings = get_settings()

_RRF_K = 60  # 标准 RRF 常数


class HybridRetriever:
    """
    检索子系统的统一入口。

    流水线
    ------
    query → DenseRetriever  ┐
                             ├─ RRF 融合 → (可选) Reranker → top-k 文档
    query → SparseRetriever ┘

    参数
    ----
    dense_retriever  : 已初始化的 DenseRetriever。
    sparse_retriever : 已初始化的 SparseRetriever（需提前构建索引）。
    reranker         : 可选的 QwenReranker（传 None 则禁用精排）。
    dense_weight     : RRF 融合前稠密分数的权重（默认 0.7）。
    sparse_weight    : RRF 融合前稀疏分数的权重（默认 0.3）。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: QwenReranker | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> None:
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
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
        完整检索流水线。

        参数
        ----
        query     : 原始查询字符串。
        dense_k   : 稠密召回数量（默认：settings.dense_top_k）。
        sparse_k  : 稀疏召回数量（默认：settings.sparse_top_k）。
        rerank_k  : 送入重排器的候选数量（默认：settings.rerank_top_k）。
        final_k   : 最终返回数量（默认：settings.final_top_k）。

        返回
        ----
        按相关性降序排列的 Document 列表。
        """
        dense_k = dense_k or settings.dense_top_k
        sparse_k = sparse_k or settings.sparse_top_k
        rerank_k = rerank_k or settings.rerank_top_k
        final_k = final_k or settings.final_top_k

        # ── 第一步：独立检索 ──────────────────────────────────────────────
        dense_results = self.dense.retrieve(query, k=dense_k)
        sparse_results = (
            self.sparse.retrieve(query, k=sparse_k)
            if self.sparse.is_ready
            else []
        )

        # ── 第二步：RRF 融合 ───────────────────────────────────────────────
        fused = self._rrf_fuse(dense_results, sparse_results)
        # 取 top rerank_k 送入重排器（无重排器则直接取 final_k）
        candidates = fused[: rerank_k if self.reranker else final_k]

        logger.info(
            f"[HybridRetriever] 稠密={len(dense_results)} "
            f"稀疏={len(sparse_results)} → 融合后={len(fused)} 候选"
        )

        # ── 第三步：精排 ───────────────────────────────────────────────────
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=final_k)
            logger.info(f"[HybridRetriever] 精排后：{len(candidates)} 条")
        else:
            candidates = candidates[:final_k]

        return candidates

    # ──────────────────────────────────────────────────────────────────────
    # RRF 实现
    # ──────────────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[tuple[Document, float]],
        sparse: list[tuple[Document, float]],
    ) -> list[Document]:
        """
        对两路排序列表执行倒数排名融合。

        文档唯一性通过 ``page_content`` 前 200 字符判定（简单高效）。
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        def _process_list(
            ranked: list[tuple[Document, float]], weight: float
        ) -> None:
            for rank, (doc, _score) in enumerate(ranked, start=1):
                key = doc.page_content[:200]  # 稳定的文档标识键
                rrf_scores[key] += weight * (1.0 / (_RRF_K + rank))
                doc_map[key] = doc

        _process_list(dense, self.dense_weight)
        _process_list(sparse, self.sparse_weight)

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]
