"""
评估数据集去重过滤脚本（Embedding 语义去重版）
──────────────────────────────────────────────────────────────────────────────
过滤规则（依次执行）：
  1. 空字段过滤   — question / answer / source 任一为空，丢弃
  2. 精确去重     — question 完全相同，保留第一条
  3. 语义去重     — 用 Embedding 模型计算 question 余弦相似度，
                    相似度 > threshold 的视为重复，保留较早出现的一条

Embedding 后端由 settings.embedding_mode 决定（local / api），
与项目其余部分保持一致。

用法
────
  # 过滤单个文件（默认输出到同目录，文件名加 _dedup 后缀）
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl

  # 指定输出路径
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl -o data/eval_dataset_ocr_dedup.jsonl

  # 调整相似度阈值（默认 0.90，越低去重越激进）
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl --threshold 0.7

  # 仅做精确去重，跳过语义去重
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl --no-semantic

  # 强制使用 API 模式 / 本地模式
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl --embedding-mode api
  python scripts/dedup_dataset.py data/eval_dataset_ocr.jsonl --embedding-mode local

  # 批量处理目录下所有 .jsonl 文件
  python scripts/dedup_dataset.py data/ --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Embedding 初始化 ──────────────────────────────────────────────────────────

def load_embedding_model(mode: str | None = None):
    from src.embeddings.qwen_embedding import QwenEmbeddings
    return QwenEmbeddings(mode=mode)


# ── 向量化 ────────────────────────────────────────────────────────────────────

def embed_questions(questions: list[str], model) -> np.ndarray:
    """批量向量化，返回 (N, D) float32 矩阵，已 L2 归一化。"""
    vecs = model.embed_documents(questions)
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ── 去重核心 ──────────────────────────────────────────────────────────────────

def dedup(
    records: list[dict],
    model,
    threshold: float = 0.90,
    semantic: bool = True,
) -> tuple[list[dict], dict]:
    """
    返回 (去重后列表, 统计信息)。
    统计 keys: total_input, empty_removed, exact_removed, semantic_removed, total_output
    """
    stats = {
        "total_input":      len(records),
        "empty_removed":    0,
        "exact_removed":    0,
        "semantic_removed": 0,
        "total_output":     0,
    }

    # ── 1. 空字段过滤 ──────────────────────────────────────────────────────
    cleaned: list[dict] = []
    for r in records:
        if not r.get("question", "").strip() \
           or not r.get("answer", "").strip() \
           or not r.get("source", "").strip():
            stats["empty_removed"] += 1
        else:
            cleaned.append(r)

    # ── 2. 精确去重 ────────────────────────────────────────────────────────
    seen: set[str] = set()
    exact_deduped: list[dict] = []
    for r in cleaned:
        q = r["question"]
        if q in seen:
            stats["exact_removed"] += 1
        else:
            seen.add(q)
            exact_deduped.append(r)

    if not semantic or len(exact_deduped) == 0:
        stats["total_output"] = len(exact_deduped)
        return exact_deduped, stats

    # ── 3. 语义去重（Embedding 余弦相似度） ───────────────────────────────
    questions = [r["question"] for r in exact_deduped]
    print(f"  向量化 {len(questions)} 条问题 ...")
    vecs = embed_questions(questions, model)        # (N, D)

    keep_mask = np.ones(len(exact_deduped), dtype=bool)
    for i in range(len(exact_deduped)):
        if not keep_mask[i]:
            continue
        # 计算 i 与所有后续条目的余弦相似度（向量已归一化，直接点积）
        sims = vecs[i + 1:] @ vecs[i]              # (N-i-1,)
        dup_indices = np.where(sims > threshold)[0] + (i + 1)
        keep_mask[dup_indices] = False

    semantic_kept = [r for r, keep in zip(exact_deduped, keep_mask) if keep]
    stats["semantic_removed"] = len(exact_deduped) - len(semantic_kept)
    stats["total_output"] = len(semantic_kept)
    return semantic_kept, stats


# ── 文件处理 ──────────────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    output_path: Path,
    model,
    threshold: float,
    semantic: bool,
) -> None:
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n{'─' * 55}")
    print(f"  文件: {input_path.name}  ({len(records)} 条)")
    print(f"{'─' * 55}")

    result, stats = dedup(records, model, threshold=threshold, semantic=semantic)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    removed = stats["empty_removed"] + stats["exact_removed"] + stats["semantic_removed"]
    print(f"  空字段过滤    : -{stats['empty_removed']}")
    print(f"  精确重复过滤  : -{stats['exact_removed']}")
    if semantic:
        print(f"  语义重复过滤  : -{stats['semantic_removed']}  (threshold={threshold})")
    print(f"  {'─' * 30}")
    print(f"  共移除        : -{removed}")
    print(f"  输出总数      : {stats['total_output']}")
    print(f"  保存至        : {output_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估数据集去重过滤（Embedding 语义去重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="输入 .jsonl 文件或目录（--batch 时为目录）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径（单文件模式）")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="语义去重余弦相似度阈值（默认 0.90）")
    parser.add_argument("--no-semantic", action="store_true",
                        help="跳过语义去重，仅做空字段 + 精确去重")
    parser.add_argument("--embedding-mode", choices=["local", "api"], default=None,
                        help="Embedding 后端（默认读取 settings.embedding_mode）")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理目录下所有 .jsonl 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic = not args.no_semantic

    # 只在需要时加载模型
    model = None
    if semantic:
        print("加载 Embedding 模型 ...")
        model = load_embedding_model(mode=args.embedding_mode)

    if args.batch:
        directory = args.input
        if not directory.is_dir():
            print(f"错误：{directory} 不是目录", file=sys.stderr)
            sys.exit(1)
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            print(f"目录 {directory} 中没有 .jsonl 文件", file=sys.stderr)
            sys.exit(1)
        for fp in files:
            out = fp.parent / (fp.stem + "_dedup.jsonl")
            process_file(fp, out, model, args.threshold, semantic)
    else:
        if not args.input.is_file():
            print(f"错误：文件不存在 {args.input}", file=sys.stderr)
            sys.exit(1)
        output_path = args.output or (
            args.input.parent / (args.input.stem + "_dedup.jsonl")
        )
        process_file(args.input, output_path, model, args.threshold, semantic)

    print(f"\n{'─' * 55}")
    print("完成。")


if __name__ == "__main__":
    main()
