"""
文档路由器
──────────────────────────────────────────────────────────────────────────────
编排双通道解析流水线：

  ┌──────────────┐
  │   输入文件   │
  └──────┬───────┘
         │ 扩展名检查
   ┌─────▼──────┐      图像文件？     ┌──────────────────┐
   │ PDF解析器  │ ── 否 ──────────── │  VisionParser    │
   └─────┬──────┘                    └──────────────────┘
         │
    逐页决策
         │
   ┌─────▼──────────────────────────────────────────────────┐
   │ needs_vision=False → 保留文本块                        │
   │ needs_vision=True  → 将该页送入 VisionParser           │
   └────────────────────────────────────────────────────────┘
         │
   合并 & 返回按页序排列的 ParsedBlock 列表

解析覆盖率统计会写入日志，调用方可借此监控解析质量。
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from .base_parser import ParsedBlock
from .pdf_parser import PDFTextParser
from .text_parser import TextParser
from .vision_parser import VisionParser


class DocumentRouter:
    """
    文档解析子系统的统一入口。

    所有文件类型均透明路由到对应解析器。
    调用方只需调用 ``route(file_path)``，即可获得扁平的
    ``ParsedBlock`` 列表，无需关心内部路由逻辑。
    """

    def __init__(self) -> None:
        self._pdf_parser = PDFTextParser()
        self._text_parser = TextParser()
        self._vision_parser = VisionParser()

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def route(self, file_path: Path) -> list[ParsedBlock]:
        """
        解析文档并返回有序的 ParsedBlock 列表。

        参数
        ----
        file_path : 文档路径（PDF 或图像格式）。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文档不存在：{file_path}")

        ext = file_path.suffix.lower()

        if self._text_parser.supports(file_path):
            logger.info(f"[Router] 文本文件 → TextParser：{file_path.name}")
            return self._text_parser.parse(file_path)

        if self._vision_parser.supports(file_path):
            logger.info(f"[Router] 图像文件 → VisionParser：{file_path.name}")
            return self._vision_parser.parse(file_path)

        if self._pdf_parser.supports(file_path):
            return self._route_pdf(file_path)

        raise ValueError(f"不支持的文件类型：{ext}")

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _route_pdf(self, file_path: Path) -> list[ParsedBlock]:
        """
        PDF 两阶段路由：
        1. 对整份文件运行 PDFTextParser。
        2. 收集标记了 needs_vision=True 的页面索引。
        3. 将这些页面通过 VisionParser 重新解析。
        4. 按页序合并两路结果。
        """
        text_blocks = self._pdf_parser.parse(file_path)

        # 按 needs_vision 标志分流
        vision_page_indices: list[int] = []
        kept_blocks: list[ParsedBlock] = []

        for blk in text_blocks:
            if blk.metadata.get("needs_vision"):
                if blk.page_num not in vision_page_indices:
                    vision_page_indices.append(blk.page_num)
            else:
                kept_blocks.append(blk)

        # 记录覆盖率统计
        total_pages = (
            max((b.page_num for b in text_blocks), default=0) + 1
        )
        vision_pages = len(vision_page_indices)
        text_pages = total_pages - vision_pages
        logger.info(
            f"[Router] {file_path.name} — "
            f"总页数={total_pages} | "
            f"文本通道={text_pages} | 视觉通道={vision_pages} "
            f"({100 * vision_pages / max(total_pages, 1):.1f}%)"
        )

        # 视觉解析阶段
        vision_blocks: list[ParsedBlock] = []
        if vision_page_indices:
            vision_blocks = self._vision_parser.parse_pdf_pages(
                file_path, sorted(vision_page_indices)
            )

        # 合并两路结果，按页码和纵向坐标排序，保持页内顺序
        all_blocks = kept_blocks + vision_blocks
        all_blocks.sort(key=lambda b: (b.page_num, b.bbox.y0 if b.bbox else 0))

        return all_blocks
