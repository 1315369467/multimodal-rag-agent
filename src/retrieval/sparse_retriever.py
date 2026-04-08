"""
稀疏检索器（BM25）
──────────────────────────────────────────────────────────────────────────────
基于 Chroma 中存储的语料库构建内存 BM25 索引。
索引在每次调用 ``build_index`` 时重建（通常在文档入库后执行一次，
之后在进程生命周期内复用缓存）。

中文文本使用 jieba 进行分词，英文按空格分词。
"""
from __future__ import annotations

import re
from typing import Optional

import jieba
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import get_settings

settings = get_settings()

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """
    支持中英文的分词器：
    • 中文使用 jieba 分词。
    • 英文按空格分词。
    • 移除标点符号，转小写。
    """
    text = _PUNCT.sub(" ", text.lower())
    # 使用 jieba 分词，自动处理中英混合文本
    tokens = jieba.cut(text)
    # 过滤空白 token
    return [token.strip() for token in tokens if token.strip()]


class SparseRetriever:
    """
    基于 BM25 的词汇检索器。

    使用示例
    --------
    retriever = SparseRetriever()
    retriever.build_index(documents)      # 入库完成后调用一次
    results = retriever.retrieve("query", k=20)
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    # ──────────────────────────────────────────────────────────────────────
    # 索引管理
    # ──────────────────────────────────────────────────────────────────────

    def build_index(self, documents: list[Document]) -> None:
        """从文档列表构建（或重建）BM25 索引。"""
        if not documents:
            logger.warning("[SparseRetriever] build_index 收到空语料库，跳过")
            return

        self._documents = documents
        corpus = [_tokenise(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        logger.info(
            f"[SparseRetriever] BM25 索引构建完成：{len(documents)} 条文档"
        )

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None

    # ──────────────────────────────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        按 BM25 分数返回 top-k 文档，分数归一化到 [0, 1]。

        参数
        ----
        query : 原始查询字符串。
        k     : 返回结果数量（默认使用 settings.sparse_top_k）。

        返回
        ----
        按归一化分数降序排列的 (Document, 分数) 列表。
        """
        if not self.is_ready:
            raise RuntimeError(
                "SparseRetriever 索引尚未构建，请先调用 build_index()。"
            )

        k = k or settings.sparse_top_k
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)  # ndarray，长度等于语料库大小

        # 按分数降序排列索引，取 top-k
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        max_score = scores[top_indices[0]] if top_indices else 1.0
        results: list[tuple[Document, float]] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                break  # 后续分数 ≤ 0，不相关，提前终止
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append((self._documents[idx], norm_score))

        logger.debug(
            f"[SparseRetriever] '{query[:60]}' → {len(results)} 条结果"
        )
        return results
