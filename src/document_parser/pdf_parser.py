"""
PDF 文本通道解析器
──────────────────────────────────────────────────────────────────────────────
使用 PyMuPDF（fitz）进行快速文本与版面提取，使用 pdfplumber 进行精确表格检测。
返回携带包围盒元数据的 ParsedBlock，供下游切分器做版面感知决策。

以下情况将页面标记为 needs_vision=True（需视觉解析）：
  • 可提取文字字符数 < settings.min_text_chars_for_text_path，或
  • 图像像素覆盖面积 > settings.image_coverage_threshold × 页面面积。
被标记的页面由 DocumentRouter 移交给 VisionParser 处理。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger

from config.settings import get_settings
from .base_parser import BaseDocumentParser, BlockType, BoundingBox, ParsedBlock

settings = get_settings()


def _norm_bbox(rect: fitz.Rect, page_w: float, page_h: float, page_num: int) -> BoundingBox:
    """将 fitz.Rect 绝对坐标转换为归一化 BoundingBox。"""
    return BoundingBox(
        x0=rect.x0 / page_w,
        y0=rect.y0 / page_h,
        x1=rect.x1 / page_w,
        y1=rect.y1 / page_h,
        page=page_num,
    )


class PDFTextParser(BaseDocumentParser):
    """
    双模式 PDF 解析器：

    文本模式 → 使用 PyMuPDF 字典级提取，保留字体/字号元数据。
    表格模式 → 使用 pdfplumber 进行单元格级文本提取。

    每页同时输出是否需要视觉回退的标志位。
    """

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def parse(self, file_path: Path) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        doc = fitz.open(str(file_path))

        with pdfplumber.open(str(file_path)) as plumber_doc:
            for page_idx, (fitz_page, plumber_page) in enumerate(
                zip(doc, plumber_doc.pages)
            ):
                page_blocks, needs_vision = self._parse_page(
                    fitz_page, plumber_page, page_idx
                )
                # 将视觉标志写入每个块的元数据，供路由器检查
                for blk in page_blocks:
                    blk.metadata["needs_vision"] = needs_vision
                blocks.extend(page_blocks)

        page_count = doc.page_count
        doc.close()
        logger.info(
            f"[PDFTextParser] {file_path.name}: {len(blocks)} blocks from "
            f"{page_count} pages"
        )
        return blocks

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _parse_page(
        self,
        fitz_page: fitz.Page,
        plumber_page: Any,
        page_idx: int,
    ) -> tuple[list[ParsedBlock], bool]:
        """返回 (当前页所有块, 是否需要视觉解析)。"""
        pw, ph = fitz_page.rect.width, fitz_page.rect.height
        blocks: list[ParsedBlock] = []

        # ── 1. 判断页面是否为图像主导/扫描件 ────────────────────────────
        needs_vision = self._is_image_dominant(fitz_page)

        if needs_vision:
            # 放入占位块，让路由器知道哪一页需要送入视觉模型
            blocks.append(
                ParsedBlock(
                    content="",
                    block_type=BlockType.FIGURE,
                    bbox=BoundingBox(0, 0, 1, 1, page_idx),
                    page_num=page_idx,
                    metadata={"needs_vision": True, "source": "pdf_image_page"},
                )
            )
            return blocks, True

        # ── 2. 使用 pdfplumber 提取表格 ───────────────────────────────────
        table_bboxes: list[BoundingBox] = []
        for table in plumber_page.extract_tables():
            # 将表格行展开为 Markdown 风格文本
            rows = [
                " | ".join(cell or "" for cell in row)
                for row in table
                if any(cell for cell in row)
            ]
            if not rows:
                continue
            table_text = "\n".join(rows)

            # 尝试从 pdfplumber 获取表格包围盒
            try:
                bbox_raw = plumber_page.find_tables()[0].bbox  # (x0, top, x1, bottom)
                bbox = BoundingBox(
                    x0=bbox_raw[0] / pw,
                    y0=bbox_raw[1] / ph,
                    x1=bbox_raw[2] / pw,
                    y1=bbox_raw[3] / ph,
                    page=page_idx,
                )
                table_bboxes.append(bbox)
            except Exception:
                bbox = BoundingBox(0, 0, 1, 1, page_idx)

            blocks.append(
                ParsedBlock(
                    content=table_text,
                    block_type=BlockType.TABLE,
                    bbox=bbox,
                    page_num=page_idx,
                    metadata={"source": "pdfplumber_table"},
                )
            )

        # ── 3. 使用 PyMuPDF 提取文本块 ────────────────────────────────────
        raw_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for raw_block in raw_dict.get("blocks", []):
            if raw_block.get("type") != 0:  # 0 = 文本块
                continue

            rect = fitz.Rect(raw_block["bbox"])
            bbox = _norm_bbox(rect, pw, ph, page_idx)

            # 跳过与已检测表格重叠度超过 30% 的文本块
            if any(self._iou(bbox, tb) > 0.3 for tb in table_bboxes):
                continue

            lines_text, font_sizes = [], []
            for line in raw_block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    line_parts.append(span.get("text", ""))
                    font_sizes.append(span.get("size", 12))
                lines_text.append("".join(line_parts))

            content = "\n".join(l for l in lines_text if l.strip())
            if not content.strip():
                continue

            avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 12
            # 字号大于 14pt 时判定为标题块
            block_type = BlockType.HEADER if avg_font > 14 else BlockType.TEXT

            blocks.append(
                ParsedBlock(
                    content=content,
                    block_type=block_type,
                    bbox=bbox,
                    page_num=page_idx,
                    metadata={
                        "avg_font_size": round(avg_font, 2),
                        "source": "pymupdf_text",
                    },
                )
            )

        return blocks, False

    # ──────────────────────────────────────────────────────────────────────

    def _is_image_dominant(self, page: fitz.Page) -> bool:
        """
        启发式判断：满足以下任一条件时认为页面图像主导：
        • 文字字符数 < 阈值，或
        • 图像面积 / 页面面积 > image_coverage_threshold。
        """
        text = page.get_text("text").strip()
        if len(text) < settings.min_text_chars_for_text_path:
            return True

        page_area = page.rect.width * page.rect.height
        image_area = 0.0
        for img_info in page.get_image_info():
            r = fitz.Rect(img_info["bbox"])
            image_area += r.width * r.height

        return (image_area / page_area) > settings.image_coverage_threshold

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        """计算两个归一化包围盒的交并比（IoU）。"""
        ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
        ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
