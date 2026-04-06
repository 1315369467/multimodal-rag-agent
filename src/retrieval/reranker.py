"""
Qwen3-Reranker（DashScope）
──────────────────────────────────────────────────────────────────────────────
使用 DashScope gte-rerank 模型实现 Cross-encoder 重排序。

DashScope 重排接口地址：
  POST /compatible-mode/v1/rerank   (OpenAI 兼容扩展)

API 接收查询 + 文档列表，返回相关性分数。
按分数降序排序后返回 top_k 个 Document。

降级策略：若重排接口不可用，记录警告并保持原始顺序（优雅降级）。
"""
from __future__ import annotations

import httpx
from langchain_core.documents import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()

_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank"
)


class QwenReranker:
    """
    使用 Qwen3-Reranker 对候选文档列表进行精排。

    参数
    ----
    model  : 重排模型名称（默认：gte-rerank）。
    top_k  : 精排后返回的文档数量。
    """

    def __init__(
        self,
        model: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.model = model or settings.reranker_model
        self.top_k = top_k or settings.final_top_k
        self._headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        对 documents 相对于 query 进行重排，返回 top_k 个最相关文档。

        参数
        ----
        query     : 原始查询字符串。
        documents : 待重排候选文档（通常 10~20 条）。
        top_k     : 覆盖实例级别的 top_k 设置。

        返回
        ----
        重排后的文档列表，最相关在前。
        """
        top_k = top_k or self.top_k
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        try:
            scores = self._call_rerank_api(query, texts)
        except Exception as exc:
            logger.warning(
                f"[Reranker] API 调用失败（{exc}），"
                "降级为保持原始顺序"
            )
            return documents[:top_k]

        # 按重排分数降序排列
        scored = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        reranked = [doc for doc, _ in scored[:top_k]]

        logger.debug(
            f"[Reranker] {len(documents)} → {len(reranked)} 条 | "
            f"最高分={scored[0][1]:.4f}" if scored else ""
        )
        return reranked

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_rerank_api(self, query: str, documents: list[str]) -> list[float]:
        """
        调用 DashScope 重排接口，返回每条文档的相关性分数。

        接口按输入顺序返回结果，此处重新对齐到原始文档顺序。
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": len(documents),  # 对全部候选打分，切片由上层完成
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(_RERANK_ENDPOINT, json=payload, headers=self._headers)
            resp.raise_for_status()

        data = resp.json()
        # DashScope 返回格式：{"results": [{"index": i, "relevance_score": f}, ...]}
        results = data.get("results", [])
        score_map = {r["index"]: r["relevance_score"] for r in results}

        # 按原始文档顺序重新对齐分数
        return [score_map.get(i, 0.0) for i in range(len(documents))]
