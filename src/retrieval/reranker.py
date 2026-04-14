"""
Qwen3-Reranker（支持本地部署 / DashScope API 两种模式）
──────────────────────────────────────────────────────────────────────────────
提供两种实现：
• QwenLocalReranker  — 本地 transformers 推理，默认 Qwen/Qwen3-Reranker-0.6B。
• QwenAPIReranker    — 通过 DashScope API 远程调用。

通过工厂函数 ``QwenReranker()`` 根据配置自动选择后端。
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


# ═══════════════════════════════════════════════════════════════════════════
# 本地模型实现
# ═══════════════════════════════════════════════════════════════════════════

class QwenLocalReranker:
    """
    基于 transformers 本地推理的 Reranker。

    参数
    ----
    model_name_or_path : HuggingFace 模型名称或本地路径。
    top_k              : 精排后返回的文档数量。
    max_length         : tokenizer 最大长度。
    instruction        : 重排指令前缀。
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        top_k: int | None = None,
        max_length: int = 4096,
        instruction: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name_or_path or settings.local_reranker_model
        self.top_k = top_k or settings.final_top_k
        self.max_length = max_length
        self.instruction = instruction or "Given the user query, retrieval the relevant passages"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left",
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, dtype=torch.float16,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model.eval()

        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")

        self._prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(self._prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(self._suffix, add_special_tokens=False)

        logger.info(f"[Reranker] 本地模型已加载: {self.model_name}")

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        top_k = top_k or self.top_k
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        scores = self._compute_scores(query, texts)

        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in scored[:top_k]]

        logger.debug(
            f"[Reranker] {len(documents)} → {len(reranked)} 条 | "
            f"最高分={scored[0][1]:.4f}" if scored else ""
        )
        return reranked

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _compute_scores(
        self, query: str, documents: list[str], batch_size: int = 4
    ) -> list[float]:
        import torch

        pairs = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]

        body_max = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        all_scores: list[float] = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start: start + batch_size]

            out = self._tokenizer(
                batch_pairs, padding=False, truncation="longest_first",
                return_attention_mask=False,
                max_length=body_max,
            )
            for i, ele in enumerate(out["input_ids"]):
                out["input_ids"][i] = self._prefix_tokens + ele + self._suffix_tokens

            out = self._tokenizer.pad(
                out, padding=True, return_tensors="pt", max_length=self.max_length
            )
            for key in out:
                out[key] = out[key].to(self._model.device)

            with torch.no_grad():
                logits = self._model(**out).logits[:, -1, :]
                true_vec = logits[:, self._token_true_id]
                false_vec = logits[:, self._token_false_id]
                stacked = torch.stack([false_vec, true_vec], dim=1)
                probs = torch.nn.functional.log_softmax(stacked, dim=1)
                batch_scores = probs[:, 1].exp().tolist()

            all_scores.extend(batch_scores)
            del out, logits, true_vec, false_vec, stacked, probs
            torch.cuda.empty_cache()

        return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# API 模型实现
# ═══════════════════════════════════════════════════════════════════════════

class QwenAPIReranker:
    """
    使用 DashScope Qwen3-Reranker API 对候选文档进行精排。

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

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
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

        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in scored[:top_k]]

        logger.debug(
            f"[Reranker] {len(documents)} → {len(reranked)} 条 | "
            f"最高分={scored[0][1]:.4f}" if scored else ""
        )
        return reranked

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_rerank_api(self, query: str, documents: list[str]) -> list[float]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": len(documents),
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(_RERANK_ENDPOINT, json=payload, headers=self._headers)
            resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        score_map = {r["index"]: r["relevance_score"] for r in results}
        return [score_map.get(i, 0.0) for i in range(len(documents))]


# ═══════════════════════════════════════════════════════════════════════════
# 工厂函数 — 根据 settings.reranker_mode 自动选择后端
# ═══════════════════════════════════════════════════════════════════════════

def QwenReranker(mode: str | None = None, **kwargs):
    """
    根据配置返回对应的 Reranker 实例。

    参数
    ----
    mode : "local" 或 "api"，为 None 时读取 settings.reranker_mode。
    **kwargs : 透传给具体实现类的构造参数。

    当 settings.enable_rerank 为 False 时直接返回 None，不加载任何模型。
    """
    if not settings.enable_rerank:
        logger.info("[Reranker] 精排已禁用（enable_rerank=False），跳过模型加载")
        return None

    mode = (mode or settings.reranker_mode).lower()
    if mode == "local":
        logger.info("[Reranker] 使用本地模型模式")
        return QwenLocalReranker(**kwargs)
    elif mode == "api":
        logger.info("[Reranker] 使用 API 模式")
        return QwenAPIReranker(**kwargs)
    else:
        raise ValueError(f"不支持的 reranker_mode: {mode!r}，请使用 'local' 或 'api'")

#  python -m src.retrieval.reranker  (从项目根目录运行)
if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    # # 将项目根目录加入 sys.path，使 config / src 等包可被找到
    # sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    model_path = "Qwen/Qwen3-Reranker-0.6B"

    query = "What is the capital of China?"
    documents = [
        # "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "人工智能是一种无监督学习方法，它可以自动从数据中提取特征，并生成模型。",
        "中国是一个国家，他的首都是北京。",
        "Python is a high-level programming language known for its readability.",
        "The Great Wall of China is one of the greatest wonders of the world.",
        "Machine learning is a subset of artificial intelligence.",
        "Beijing is the political and cultural center of China.",
    ]

    # ── 加载模型 ──────────────────────────────────────────────────────────
    print(f"正在加载模型: {model_path}")
    t0 = time.perf_counter()
    reranker = QwenLocalReranker(model_name_or_path=model_path, top_k=5)
    load_time = time.perf_counter() - t0
    print(f"模型加载耗时: {load_time:.2f}s\n")

    # ── 重排测试 ──────────────────────────────────────────────────────────
    print(f"Query: {query}")
    print(f"候选文档数: {len(documents)}")

    langchain_docs = [Document(page_content=text) for text in documents]

    t0 = time.perf_counter()
    ranked = reranker.rerank(query, langchain_docs, top_k=len(documents))
    rerank_time = time.perf_counter() - t0
    print(f"重排耗时: {rerank_time * 1000:.1f}ms\n")

    print("重排结果 (按相关性降序):")
    scores = reranker._compute_scores(query, [doc.page_content for doc in ranked])
    for i, (doc, score) in enumerate(zip(ranked, scores)):
        text_preview = doc.page_content[:60]
        print(f"  [{i + 1}] score={score:.4f}  {text_preview}")

    # ── 多 Query 吞吐量测试 ──────────────────────────────────────────────
    test_queries = [
        "What is the capital of China?",
        "Explain gravity",
        "What is machine learning?",
        "Tell me about the Great Wall",
    ]
    print(f"\n吞吐量测试: {len(test_queries)} 个 query × {len(documents)} 条文档 ...")
    t0 = time.perf_counter()
    for q in test_queries:
        _ = reranker.rerank(q, langchain_docs, top_k=3)
    throughput_time = time.perf_counter() - t0
    qps = len(test_queries) / throughput_time
    print(f"  耗时: {throughput_time:.2f}s  |  {qps:.1f} query/s")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"  模型加载       : {load_time:.2f}s")
    print(f"  单次重排 ({len(documents)}条) : {rerank_time * 1000:.1f}ms")
    print(f"  吞吐量         : {qps:.1f} query/s ({len(documents)} 候选/query)")
