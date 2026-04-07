"""
RAG 检索评估脚本
──────────────────────────────────────────────────────────────────────────────
在带标注的问答数据集上评估检索流水线。

支持两种模式
────────────
1. 离线模式（默认）：从本地语料库目录加载文档，仅使用 BM25 稀疏检索。
   无需 API Key、无需向量数据库，开箱即用。
2. 在线模式（--online）：使用完整的 Dense + Sparse + Reranker 流水线，
   需要 DashScope API Key 和已构建的 Chroma 向量库。

计算指标
────────
  Recall@K    — 正确来源出现在 top-K 结果中的查询比例。
  MRR@K       — 平均倒数排名（Mean Reciprocal Rank）@K。
  Precision@K — top-K 结果中正确来源的比例。
  NDCG@K      — 归一化折损累计增益（衡量排序质量）。
  Hit Rate@K  — 至少命中一条相关文档的查询比例（同 Recall，单答案时等价）。

数据集格式（JSON Lines，每行一条查询）：
  {"question": "...", "answer": "...", "source": "doc.txt", "page": 0}

语料库格式：
  一个目录下的 .txt / .md 文件，文件名即 source 标识。

用法
────
  # 离线模式（默认，无需 API）
  python scripts/evaluate.py

  # 指定数据集和语料库
  python scripts/evaluate.py --dataset data/eval_dataset.jsonl --corpus data/eval_corpus

  # 在线模式（完整 RAG 流水线）
  python scripts/evaluate.py --online --dataset data/eval_dataset.jsonl

  # 多个 K 值对比
  python scripts/evaluate.py --k 3 5 10

  # 消融实验：仅 Dense / 仅 Sparse
  python scripts/evaluate.py --online --ablation dense
  python scripts/evaluate.py --online --ablation sparse
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from src.retrieval.sparse_retriever import SparseRetriever


# ── 指标函数 ──────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_sources: list[str], ground_truth: str, k: int
) -> float:
    """若正确来源出现在 top-k 中则返回 1，否则返回 0。"""
    return float(ground_truth in retrieved_sources[:k])


def reciprocal_rank(
    retrieved_sources: list[str], ground_truth: str, k: int
) -> float:
    """找到正确来源则返回 1/rank，否则返回 0。仅在 top-k 范围内计算。"""
    for rank, src in enumerate(retrieved_sources[:k], start=1):
        if src == ground_truth:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_sources: list[str], ground_truth: str, k: int
) -> float:
    """top-k 中命中正确来源的比例。"""
    hits = sum(1 for s in retrieved_sources[:k] if s == ground_truth)
    return hits / k


def ndcg_at_k(
    retrieved_sources: list[str], ground_truth: str, k: int
) -> float:
    """
    NDCG@K（单一正确答案版本）。
    仅计算首次命中的排名位置，后续重复命中不计入 DCG。
    """
    dcg = 0.0
    found = False
    for rank, src in enumerate(retrieved_sources[:k], start=1):
        if src == ground_truth and not found:
            dcg += 1.0 / math.log2(rank + 1)
            found = True
    idcg = 1.0 / math.log2(2)  # 理想情况：正确来源在第1位
    return dcg / idcg if idcg > 0 else 0.0


# ── 离线语料库加载 ────────────────────────────────────────────────────────

def load_corpus_from_directory(corpus_dir: Path) -> dict:
    """
    从目录加载 .txt/.md 文件，按章节切分为 LangChain Document。

    使用基于 Markdown 标题（## / ###）的章节切分，而非 token 估算，
    确保中文文档也能产生合理粒度的文档块。

    返回
    ----
    {
        "documents": list[Document],  # 全部切分后的文档块
        "source_map": dict,           # source_name -> 原始内容
    }
    """
    import re
    from langchain_core.documents import Document

    all_docs = []
    source_map = {}

    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.txt")) + sorted(corpus_dir.glob("*.md"))

    if not files:
        logger.error(f"语料库目录 {corpus_dir} 中无 .txt/.md 文件")
        sys.exit(1)

    for file_path in files:
        logger.info(f"[Corpus] 解析：{file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        source_map[file_path.name] = text

        # 按二级/三级标题切分（### 优先于 ##）
        sections = re.split(r"\n(?=#{2,3}\s)", text)
        for section in sections:
            section = section.strip()
            if not section or len(section) < 20:
                continue
            all_docs.append(
                Document(
                    page_content=section,
                    metadata={"source": file_path.name, "page_num": 0},
                )
            )

    logger.info(
        f"[Corpus] 共加载 {len(files)} 个文件，"
        f"切分为 {len(all_docs)} 个文档块"
    )
    return {"documents": all_docs, "source_map": source_map}


# ── 在线检索器初始化 ──────────────────────────────────────────────────────

def init_online_retriever(ablation: str | None = None):
    """初始化完整 RAG 检索流水线（需要 API Key）。"""
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import QwenReranker
    from src.vectorstore.chroma_store import ChromaVectorStore

    store = ChromaVectorStore()
    dense = DenseRetriever(store)
    sparse = SparseRetriever()

    corpus = store.get_all_documents(limit=100_000)
    if corpus:
        sparse.build_index(corpus)

    if ablation == "dense":
        # 仅稠密检索：sparse 权重设为 0
        reranker = None
        return HybridRetriever(dense, sparse, reranker, dense_weight=1.0, sparse_weight=0.0)
    elif ablation == "sparse":
        # 仅稀疏检索：dense 权重设为 0
        reranker = None
        return HybridRetriever(dense, sparse, reranker, dense_weight=0.0, sparse_weight=1.0)
    else:
        reranker = QwenReranker()
        return HybridRetriever(dense, sparse, reranker)


# ── 离线 BM25 检索 ────────────────────────────────────────────────────────

def retrieve_offline(
    sparse: SparseRetriever, query: str, k: int
) -> list[dict]:
    """使用 BM25 检索，返回 top-k 结果的来源信息。"""
    results = sparse.retrieve(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", ""),
            "page_num": doc.metadata.get("page_num", 0),
            "content_preview": doc.page_content[:80],
            "score": score,
        }
        for doc, score in results
    ]


def retrieve_online(
    retriever, query: str, k: int
) -> list[dict]:
    """使用完整 Hybrid 检索流水线。"""
    docs = retriever.retrieve(query, dense_k=k * 2, sparse_k=k * 2, final_k=k)
    return [
        {
            "source": doc.metadata.get("source", ""),
            "page_num": doc.metadata.get("page_num", 0),
            "content_preview": doc.page_content[:80],
            "score": 0.0,
        }
        for doc in docs
    ]


# ── 评估核心 ─────────────────────────────────────────────────────────────

def evaluate(
    queries: list[dict],
    retrieve_fn,
    k_values: list[int],
) -> dict:
    """
    对查询集合执行检索并计算全部指标。

    参数
    ----
    queries     : 评估查询列表。
    retrieve_fn : 接受 (query, k) 返回 list[dict] 的检索函数。
    k_values    : 要计算的 K 值列表。

    返回
    ----
    包含 summary（按 K 值分组的指标）和 per_query 详细结果的字典。
    """
    max_k = max(k_values)

    # 按来源分组统计
    category_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_query_results = []

    # 按 K 值分组的全局指标
    metrics_by_k: dict[int, dict[str, list[float]]] = {
        k: defaultdict(list) for k in k_values
    }

    for item in tqdm(queries, desc="评估中", unit="条"):
        question = item["question"]
        gt_source = item.get("source", "")
        category = item.get("category", "unknown")

        # 一次检索取最大 K 即可
        results = retrieve_fn(question, max_k)
        retrieved_sources = [r["source"] for r in results]

        query_result = {
            "question": question,
            "ground_truth_source": gt_source,
            "category": category,
            "retrieved_top3": [r["source"] for r in results[:3]],
            "metrics": {},
        }

        for k in k_values:
            r_k = recall_at_k(retrieved_sources, gt_source, k)
            mrr = reciprocal_rank(retrieved_sources, gt_source, k)
            p_k = precision_at_k(retrieved_sources, gt_source, k)
            n_k = ndcg_at_k(retrieved_sources, gt_source, k)

            metrics_by_k[k]["recall"].append(r_k)
            metrics_by_k[k]["mrr"].append(mrr)
            metrics_by_k[k]["precision"].append(p_k)
            metrics_by_k[k]["ndcg"].append(n_k)

            # 按分类统计
            category_scores[category][f"recall@{k}"].append(r_k)
            category_scores[category][f"mrr@{k}"].append(mrr)

            query_result["metrics"][f"recall@{k}"] = r_k
            query_result["metrics"][f"mrr@{k}"] = mrr
            query_result["metrics"][f"ndcg@{k}"] = n_k

        per_query_results.append(query_result)

    # 汇总
    summary = {}
    for k in k_values:
        n = len(queries)
        summary[f"Recall@{k}"] = sum(metrics_by_k[k]["recall"]) / n
        summary[f"MRR@{k}"] = sum(metrics_by_k[k]["mrr"]) / n
        summary[f"Precision@{k}"] = sum(metrics_by_k[k]["precision"]) / n
        summary[f"NDCG@{k}"] = sum(metrics_by_k[k]["ndcg"]) / n

    # 分类汇总
    category_summary = {}
    for cat, scores in category_scores.items():
        cat_summary = {}
        for metric_name, values in scores.items():
            cat_summary[metric_name] = sum(values) / len(values) if values else 0.0
        cat_summary["count"] = len(scores.get(f"recall@{k_values[0]}", []))
        category_summary[cat] = cat_summary

    # 失败案例分析
    failed_queries = [
        q for q in per_query_results if q["metrics"].get(f"recall@{max_k}", 0) == 0
    ]

    return {
        "summary": summary,
        "category_summary": category_summary,
        "per_query": per_query_results,
        "failed_queries": failed_queries,
        "num_queries": len(queries),
    }


# ── 结果展示 ──────────────────────────────────────────────────────────────

def print_results(output: dict, k_values: list[int], mode: str, elapsed: float) -> None:
    """格式化打印评估结果。"""
    summary = output["summary"]
    n = output["num_queries"]

    print("\n" + "=" * 60)
    print(f"  RAG 检索评估报告 （{mode}模式）")
    print("=" * 60)
    print(f"  评估查询数  : {n}")
    print(f"  评估耗时    : {elapsed:.1f}s")
    print()

    # 主指标表格
    print("  ┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │      K      │ Recall   │ MRR      │ Precision│ NDCG     │")
    print("  ├─────────────┼──────────┼──────────┼──────────┼──────────┤")
    for k in k_values:
        r = summary[f"Recall@{k}"]
        m = summary[f"MRR@{k}"]
        p = summary[f"Precision@{k}"]
        nd = summary[f"NDCG@{k}"]
        print(f"  │  K = {k:<6} │ {r:>7.1%} │ {m:>7.4f} │ {p:>7.1%} │ {nd:>7.4f} │")
    print("  └─────────────┴──────────┴──────────┴──────────┴──────────┘")

    # 分类细分
    cat_summary = output.get("category_summary", {})
    if cat_summary:
        print(f"\n  分类细分（K={k_values[-1]}）:")
        print("  ┌─────────────┬──────┬──────────┬──────────┐")
        print("  │ 分类        │ 数量 │ Recall   │ MRR      │")
        print("  ├─────────────┼──────┼──────────┼──────────┤")
        k_last = k_values[-1]
        for cat, data in sorted(cat_summary.items()):
            cnt = data.get("count", 0)
            recall = data.get(f"recall@{k_last}", 0)
            mrr = data.get(f"mrr@{k_last}", 0)
            print(
                f"  │ {cat:<11} │ {cnt:>4} │ {recall:>7.1%} │ {mrr:>7.4f} │"
            )
        print("  └─────────────┴──────┴──────────┴──────────┘")

    # 失败案例
    failed = output.get("failed_queries", [])
    if failed:
        print(f"\n  未命中查询（{len(failed)} 条）:")
        for i, q in enumerate(failed[:10], 1):
            print(f"    {i}. [{q['ground_truth_source']}] {q['question']}")
            if q.get("retrieved_top3"):
                print(f"       → 实际检索到: {q['retrieved_top3']}")
        if len(failed) > 10:
            print(f"    ... 共 {len(failed)} 条未命中")

    print("\n" + "=" * 60)


# ── 主函数 ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 RAG 检索流水线（支持离线/在线模式）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval_dataset.jsonl"),
        help="JSONL 评估数据集路径（默认：data/eval_dataset.jsonl）",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/eval_corpus"),
        help="离线模式语料库目录（默认：data/eval_corpus）",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="指标计算的 Top-K 值列表（默认：3 5 10）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="结果输出 JSON 路径（默认：eval_results.json）",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="使用在线模式（完整 Dense+Sparse+Reranker 流水线）",
    )
    parser.add_argument(
        "--ablation",
        choices=["dense", "sparse"],
        default=None,
        help="消融实验：仅使用 dense 或 sparse 检索（仅在线模式有效）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="离线模式的切分大小（默认：256 tokens）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 加载数据集 ────────────────────────────────────────────────────────
    if not args.dataset.exists():
        logger.error(f"数据集不存在：{args.dataset}")
        sys.exit(1)

    queries = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    logger.info(f"已加载 {len(queries)} 条评估查询")

    # ── 初始化检索器 ──────────────────────────────────────────────────────
    if args.online:
        mode = "在线"
        logger.info("使用在线模式（完整 RAG 流水线）")
        retriever = init_online_retriever(args.ablation)

        def retrieve_fn(query: str, k: int):
            return retrieve_online(retriever, query, k)
    else:
        mode = "离线/BM25"
        logger.info("使用离线模式（BM25 稀疏检索）")

        if not args.corpus.exists():
            logger.error(f"语料库目录不存在：{args.corpus}")
            sys.exit(1)

        corpus_data = load_corpus_from_directory(args.corpus)
        sparse = SparseRetriever()
        sparse.build_index(corpus_data["documents"])

        def retrieve_fn(query: str, k: int):
            return retrieve_offline(sparse, query, k)

    # ── 执行评估 ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    output = evaluate(queries, retrieve_fn, sorted(args.k))
    elapsed = time.perf_counter() - t0

    # ── 展示结果 ──────────────────────────────────────────────────────────
    ablation_label = f"（消融：仅{args.ablation}）" if args.ablation else ""
    print_results(output, sorted(args.k), f"{mode}{ablation_label}", elapsed)

    # ── 保存详细结果 ──────────────────────────────────────────────────────
    output["meta"] = {
        "mode": mode,
        "dataset": str(args.dataset),
        "k_values": sorted(args.k),
        "num_queries": len(queries),
        "elapsed_seconds": round(elapsed, 2),
        "ablation": args.ablation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  详细结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
