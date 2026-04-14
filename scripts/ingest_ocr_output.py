"""
OCR 解析结果入库 CLI 脚本
──────────────────────────────────────────────────────────────────────────────
将 MinerU OCR 解析生成的 Markdown 文件批量写入 Chroma 向量数据库。

OCR 输出目录结构（由 mineru_ocr.py 生成）：
    {ocr_dir}/
      {文档名}/
        {backend}/
          {文档名}.md      ← 入库目标
          images/          ← 提取出的图片（暂不入库）

用法
────
# 入库默认目录 data/ocr_output 下所有文档
python scripts/ingest_ocr_output.py

# 指定其他 OCR 输出目录
python scripts/ingest_ocr_output.py --ocr-dir ./my_ocr_results

# 先清空集合再入库
python scripts/ingest_ocr_output.py --reset

# 指定切分策略（fixed / sentence / paragraph / markdown / recursive / semantic）
python scripts/ingest_ocr_output.py --chunk-strategy markdown
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from config.settings import get_settings
from src.document_parser.chunker import SUPPORTED_STRATEGIES, chunk_blocks
from src.document_parser.text_parser import TextParser
from src.retrieval.sparse_retriever import SparseRetriever
from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()

DEFAULT_OCR_DIR = Path(__file__).resolve().parents[1] / "data" / "ocr_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 MinerU OCR 解析结果（Markdown）入库到多模态 RAG 知识库。"
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=DEFAULT_OCR_DIR,
        help=f"OCR 输出根目录（默认：{DEFAULT_OCR_DIR}）。",
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


def collect_md_files(ocr_dir: Path) -> list[tuple[Path, str]]:
    """
    扫描 OCR 输出目录，返回 (md文件路径, 文档名) 列表。

    目录结构：{ocr_dir}/{文档名}/{backend}/{文档名}.md
    """
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR 输出目录不存在：{ocr_dir}")

    results: list[tuple[Path, str]] = []

    for doc_dir in sorted(ocr_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_name = doc_dir.name
        # 在所有 backend 子目录中查找 .md 文件
        md_files = sorted(doc_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"[扫描] {doc_name}：未找到 Markdown 文件，跳过")
            continue
        # 若同一文档有多个 .md（多个 backend），取第一个
        md_path = md_files[0]
        if len(md_files) > 1:
            logger.warning(
                f"[扫描] {doc_name}：发现 {len(md_files)} 个 Markdown 文件，"
                f"使用：{md_path.relative_to(ocr_dir)}"
            )
        results.append((md_path, doc_name))

    return results


def main() -> None:
    _start_time = time.time()
    args = parse_args()

    # ── 初始化 ────────────────────────────────────────────────────────────
    store = ChromaVectorStore()
    if args.reset:
        logger.warning("按要求重置集合……")
        store.delete_collection()

    text_parser = TextParser()
    sparse = SparseRetriever()

    # ── 收集文件 ──────────────────────────────────────────────────────────
    try:
        md_files = collect_md_files(args.ocr_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not md_files:
        logger.error(f"在 {args.ocr_dir} 中未找到任何 Markdown 文件，退出。")
        sys.exit(1)

    logger.info(f"发现 {len(md_files)} 个 OCR 文档，切分策略：{args.chunk_strategy}")

    # ── 处理循环 ──────────────────────────────────────────────────────────
    total_chunks = 0
    failed: list[str] = []

    for md_path, doc_name in tqdm(md_files, desc="入库中", unit="文档"):
        try:
            blocks = text_parser.parse(md_path)
            chunks = chunk_blocks(
                blocks=blocks,
                strategy=args.chunk_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                source_name=doc_name,
            )

            if chunks:
                store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(
                    f"✓ {doc_name}：{len(blocks)} blocks → {len(chunks)} chunks"
                    f"（来源：{md_path.relative_to(args.ocr_dir)}）"
                )
            else:
                logger.warning(f"⚠ {doc_name}：未生成任何 chunk")

        except Exception as exc:
            logger.error(f"✗ {doc_name}：{exc}")
            failed.append(doc_name)

    # ── 构建 BM25 索引 ────────────────────────────────────────────────────
    logger.info("正在构建 BM25 稀疏索引……")
    corpus = store.get_all_documents(limit=100_000)
    if corpus:
        sparse.build_index(corpus)

    # ── 汇总输出 ──────────────────────────────────────────────────────────
    elapsed = time.time() - _start_time
    print("\n" + "=" * 60)
    print(f"  OCR 输出目录 : {args.ocr_dir}")
    print(f"  已处理文档数 : {len(md_files) - len(failed)} / {len(md_files)}")
    print(f"  生成 chunk 数 : {total_chunks}")
    print(f"  库中总文档数 : {store.count()}")
    if failed:
        print(f"  失败文档数   : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"  总耗时       : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
