"""
文档批量入库 CLI 脚本
──────────────────────────────────────────────────────────────────────────────
从目录或指定文件列表中批量解析文档，并将所有 chunk 写入 Chroma。
适合初次构建语料库时使用。

用法
────
# 入库目录下所有 PDF
python scripts/ingest_documents.py --input-dir ./data/eval_corpus

# 入库指定文件
python scripts/ingest_documents.py --files report.pdf scan.png table.pdf

# 先清空集合再入库
python scripts/ingest_documents.py --input-dir ./data/eval_corpus --reset

# 指定切分策略（fixed / sentence / paragraph / markdown / recursive / semantic）
python scripts/ingest_documents.py --input-dir ./data/eval_corpus --chunk-strategy recursive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 以脚本方式运行时，将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from config.settings import get_settings
from src.document_parser.chunker import SUPPORTED_STRATEGIES, chunk_blocks
from src.document_parser.router import DocumentRouter
from src.retrieval.sparse_retriever import SparseRetriever
from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".txt", ".md", ".markdown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将文档入库到多模态 RAG 知识库。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        type=Path,
        help="递归扫描该目录下所有受支持的文档。",
    )
    group.add_argument(
        "--files",
        nargs="+",
        # default="./data/documents/CodeCV简历.pdf",
        type=Path,
        help="显式指定要入库的文件路径列表。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="入库前先清空现有集合。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"chunk 大小（近似 token 数，默认：{settings.chunk_size}）。",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"chunk 重叠量（近似 token 数，默认：{settings.chunk_overlap}）。",
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default=settings.chunk_strategy,
        choices=sorted(SUPPORTED_STRATEGIES),
        help=f"切分策略（默认：{settings.chunk_strategy}）。",
    )
    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        files = [
            f
            for f in args.input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        logger.info(f"在 {args.input_dir} 中发现 {len(files)} 个文件")
    else:
        files = [p for p in args.files if p.exists()]
        missing = [p for p in args.files if not p.exists()]
        if missing:
            logger.warning(f"跳过不存在的文件：{[str(m) for m in missing]}")
    return sorted(files)


def main() -> None:
    args = parse_args()

    # ── 初始化 ────────────────────────────────────────────────────────────
    store = ChromaVectorStore()
    if args.reset:
        logger.warning("按要求重置集合……")
        store.delete_collection()

    router = DocumentRouter()
    sparse = SparseRetriever()

    files = collect_files(args)
    if not files:
        logger.error("没有可处理的文件，退出。")
        sys.exit(1)

    # ── 处理循环 ──────────────────────────────────────────────────────────
    total_chunks = 0
    failed: list[str] = []

    logger.info(f"切分策略：{args.chunk_strategy}")

    for file_path in tqdm(files, desc="入库中", unit="文件"):
        try:
            blocks = router.route(file_path)
            chunks = chunk_blocks(
                blocks=blocks,
                strategy=args.chunk_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                source_name=file_path.name,
            )

            if chunks:
                store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(
                    f"✓ {file_path.name}：{len(blocks)} blocks → {len(chunks)} chunks"
                )
            else:
                logger.warning(f"⚠ {file_path.name}：未生成任何 chunk")

        except Exception as exc:
            logger.error(f"✗ {file_path.name}：{exc}")
            failed.append(str(file_path))

    # ── 构建 BM25 索引 ────────────────────────────────────────────────────
    logger.info("正在构建 BM25 稀疏索引……")
    corpus = store.get_all_documents(limit=100_000)
    if corpus:
        sparse.build_index(corpus)

    # ── 汇总输出 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  已处理文件数 : {len(files) - len(failed)} / {len(files)}")
    print(f"  生成 chunk 数 : {total_chunks}")
    print(f"  库中总文档数 : {store.count()}")
    if failed:
        print(f"  失败文件数   : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
