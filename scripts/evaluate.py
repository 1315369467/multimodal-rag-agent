"""
RAG 检索评估脚本
──────────────────────────────────────────────────────────────────────────────
在带标注的问答数据集上评估检索流水线。

计算指标
────────
  Recall@K    — 正确来源出现在 top-K 结果中的查询比例。
  MRR@K       — 平均倒数排名（Mean Reciprocal Rank）@K。
  Precision@K — top-K 结果中正确来源的比例（单答案版本）。

数据集格式（JSON Lines，每行一条查询）：
  {"question": "...", "answer": "...", "source": "doc.pdf", "page": 3}

用法
────
  python scripts/evaluate.py \
      --dataset ./data/eval_dataset.jsonl \
      --k 10 \
      --output ./results/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from config.settings import get_settings
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import QwenReranker
from src.retrieval.sparse_retriever import SparseRetriever
from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()


# ── 指标函数 ──────────────────────────────────────────────────────────────

def recall_at_k(retrieved_sources: list[tuple[str, int]], ground_truth: tuple[str, int], k: int) -> float:
    """若正确来源 (source, page) 出现在 top-k 中则返回 1，否则返回 0。"""
    return float(ground_truth in retrieved_sources[:k])


def reciprocal_rank(retrieved_sources: list[tuple[str, int]], ground_truth: tuple[str, int]) -> float:
    """找到正确来源则返回 1/rank，否则返回 0。"""
    for rank, src in enumerate(retrieved_sources, start=1):
        if src == ground_truth:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_sources: list[tuple[str, int]], ground_truth: tuple[str, int], k: int) -> float:
    """top-k 中命中正确来源的比例（单答案版本）。"""
    hits = sum(1 for s in retrieved_sources[:k] if s == ground_truth)
    return hits / k


# ── 主函数 ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 RAG 检索流水线。")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL 评估数据集路径。")
    parser.add_argument("--k", type=int, default=10, help="指标计算的 Top-K（默认：10）。")
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"), help="结果输出 JSON 路径。")
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排器（用于消融实验）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        logger.error(f"数据集不存在：{args.dataset}")
        sys.exit(1)

    # ── 加载数据集 ────────────────────────────────────────────────────────
    queries = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    logger.info(f"已加载 {len(queries)} 条评估查询")

    # ── 初始化检索器 ──────────────────────────────────────────────────────
    store = ChromaVectorStore()
    dense = DenseRetriever(store)
    sparse = SparseRetriever()
    corpus = store.get_all_documents(limit=100_000)
    if corpus:
        sparse.build_index(corpus)

    reranker = None if args.no_rerank else QwenReranker()
    retriever = HybridRetriever(dense, sparse, reranker)

    # ── 评估循环 ──────────────────────────────────────────────────────────
    recall_scores, mrr_scores, precision_scores = [], [], []
    per_query_results = []

    for item in tqdm(queries, desc="评估中", unit="条"):
        question = item["question"]
        gt_source = item.get("source", "")
        gt_page = int(item.get("page", -1))
        ground_truth = (gt_source, gt_page)

        docs = retriever.retrieve(
            question,
            dense_k=args.k * 2,
            sparse_k=args.k * 2,
            rerank_k=args.k,
            final_k=args.k,
        )

        retrieved_sources = [
            (doc.metadata.get("source", ""), int(doc.metadata.get("page_num", -1)))
            for doc in docs
        ]

        r_k = recall_at_k(retrieved_sources, ground_truth, args.k)
        mrr = reciprocal_rank(retrieved_sources, ground_truth)
        p_k = precision_at_k(retrieved_sources, ground_truth, args.k)

        recall_scores.append(r_k)
        mrr_scores.append(mrr)
        precision_scores.append(p_k)

        per_query_results.append(
            {
                "question": question,
                "ground_truth": {"source": gt_source, "page": gt_page},
                "recall": r_k,
                "mrr": mrr,
                "precision": p_k,
                "retrieved": [{"source": s, "page": p} for s, p in retrieved_sources],
            }
        )

    # ── 汇总指标 ──────────────────────────────────────────────────────────
    n = len(queries)
    summary = {
        f"Recall@{args.k}": sum(recall_scores) / n,
        f"MRR@{args.k}": sum(mrr_scores) / n,
        f"Precision@{args.k}": sum(precision_scores) / n,
        "num_queries": n,
        "k": args.k,
        "reranker_enabled": not args.no_rerank,
    }

    # ── 输出结果 ──────────────────────────────────────────────────────────
    output = {"summary": summary, "per_query": per_query_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print("\n" + "=" * 50)
    print("  评估结果")
    print("=" * 50)
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"  {metric:<20}: {value:.4f}")
        else:
            print(f"  {metric:<20}: {value}")
    print("=" * 50)
    print(f"\n详细结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
