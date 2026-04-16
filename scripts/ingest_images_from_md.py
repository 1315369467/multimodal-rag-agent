"""
从 Markdown 文件中提取本地图片并向量化入库
──────────────────────────────────────────────────────────────────────────────
扫描 .md / .markdown 文件，找出所有 ![alt](local/path.png) 形式的引用，
使用 Qwen3-Embedding-VL-2B 将图片向量化，存入独立的图片向量集合。

用法
────
# 扫描目录下所有 md 文件
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output

# 指定单个 md 文件
python scripts/ingest_images_from_md.py --files ./data/ocr_output/report.md

# 重置图片集合后重新入库
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output --reset

注意
────
• 只处理本地相对/绝对路径的图片（http(s):// 开头的 URL 自动跳过）。
• 图片路径相对于所在 md 文件的目录进行解析。
• 重复运行是幂等的（相同图片路径不会重复写入）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from src.document_parser.text_parser import TextParser
from src.document_parser.base_parser import BlockType
from src.vectorstore.image_store import ImageChromaStore

SUPPORTED_EXTENSIONS = {".md", ".markdown"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Markdown 文件中提取本地图片并向量化入库。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir", type=Path,
        help="递归扫描该目录下所有 .md / .markdown 文件。",
    )
    group.add_argument(
        "--files", nargs="+", type=Path,
        help="显式指定要处理的 Markdown 文件列表。",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="入库前先清空现有图片集合。",
    )
    return parser.parse_args()


def collect_md_files(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        files = [
            f for f in args.input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        logger.info(f"在 {args.input_dir} 中发现 {len(files)} 个 Markdown 文件")
    else:
        files = [p for p in args.files if p.exists()]
        missing = [p for p in args.files if not p.exists()]
        if missing:
            logger.warning(f"跳过不存在的文件：{[str(m) for m in missing]}")
    return sorted(files)


def extract_local_images(
    md_file: Path, parser: TextParser
) -> list[tuple[Path, dict]]:
    """
    解析 md 文件，返回所有本地图片的 (绝对路径, 元数据) 列表。

    元数据包含：
      source     — md 文件名
      source_abs — md 文件绝对路径
      alt_text   — 图片 alt 文本
      caption    — 图注（若有）
    """
    blocks = parser.parse(md_file)
    results: list[tuple[Path, dict]] = []

    for block in blocks:
        if block.block_type != BlockType.FIGURE:
            continue

        image_url = block.metadata.get("image_url", "")
        if not image_url:
            continue

        # 跳过网络 URL
        if image_url.startswith(("http://", "https://", "//", "data:")):
            continue

        # 相对路径 → 绝对路径（相对于 md 文件所在目录）
        image_path = (md_file.parent / image_url).resolve()

        # 检查文件是否存在且后缀合法
        if not image_path.exists():
            logger.warning(f"[跳过] 图片不存在: {image_path}  (引用于 {md_file.name})")
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.debug(f"[跳过] 不支持的图片格式: {image_path.suffix}")
            continue

        meta = {
            "source": md_file.name,
            "source_abs": str(md_file.resolve()),
            "alt_text": block.metadata.get("alt_text", ""),
            "caption": block.metadata.get("caption", ""),
            "image_url_raw": image_url,  # 原始相对路径，方便调试
        }
        results.append((image_path, meta))

    return results


def main() -> None:
    t_start = time.time()
    args = parse_args()

    # ── 初始化 ────────────────────────────────────────────────────────────
    logger.info("初始化 ImageChromaStore（会加载 VL 嵌入模型，可能需要一些时间）……")
    store = ImageChromaStore()

    if args.reset:
        logger.warning("按要求重置图片集合……")
        store.delete_collection()

    parser = TextParser()
    md_files = collect_md_files(args)

    if not md_files:
        logger.error("没有可处理的 Markdown 文件，退出。")
        sys.exit(1)

    # ── 收集所有图片 ──────────────────────────────────────────────────────
    all_images: list[tuple[Path, dict]] = []
    for md_file in md_files:
        pairs = extract_local_images(md_file, parser)
        logger.info(f"  {md_file.name}: 发现 {len(pairs)} 张本地图片")
        all_images.extend(pairs)

    # 去重（同一图片可能被多个 md 引用）
    seen: set[str] = set()
    unique_images: list[tuple[Path, dict]] = []
    for path, meta in all_images:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_images.append((path, meta))

    logger.info(f"共发现 {len(unique_images)} 张唯一本地图片，开始向量化入库……")

    if not unique_images:
        logger.warning("未发现任何本地图片，退出。")
        sys.exit(0)

    # ── 批量入库 ──────────────────────────────────────────────────────────
    paths = [p for p, _ in unique_images]
    metas = [m for _, m in unique_images]

    total_ok = 0
    failed: list[str] = []

    # 分批处理（每批 20 张，方便展示进度）
    batch_size = 20
    for i in tqdm(range(0, len(paths), batch_size), desc="图片入库", unit="批"):
        batch_paths = paths[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        try:
            store.add_images(batch_paths, batch_metas)
            total_ok += len(batch_paths)
        except Exception as exc:
            logger.error(f"第 {i // batch_size + 1} 批入库失败: {exc}")
            failed.extend(str(p) for p in batch_paths)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  已处理图片数  : {total_ok} / {len(unique_images)}")
    print(f"  图片库总数量  : {store.count()}")
    if failed:
        print(f"  失败图片数    : {len(failed)}")
        for f in failed[:10]:
            print(f"    - {f}")
    print(f"  总耗时        : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
