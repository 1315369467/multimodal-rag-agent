"""
文档路由器
──────────────────────────────────────────────────────────────────────────────
将输入文件分发至对应解析器：

  .md / .markdown / .txt  →  TextParser（结构化 Markdown 解析）
  图像格式                 →  VisionParser（Qwen-VL 多模态解析）

所有待切分文档均为 Markdown 格式；TextParser 会将标题、表格、图片、
代码块分别映射为带类型标注的 ParsedBlock，供下游 StructureAwareChunker 使用。
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from .base_parser import ParsedBlock
from .text_parser import TextParser
from .vision_parser import VisionParser


class DocumentRouter:
    """
    文档解析子系统的统一入口。

    调用方只需调用 ``route(file_path)``，即可获得扁平的
    ``ParsedBlock`` 列表，无需关心内部路由逻辑。
    """

    def __init__(self) -> None:
        self._text_parser = TextParser()
        self._vision_parser = VisionParser()

    def route(self, file_path: Path) -> list[ParsedBlock]:
        """
        解析文档并返回有序的 ParsedBlock 列表。

        参数
        ----
        file_path : 文档路径（Markdown、纯文本或图像格式）。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文档不存在：{file_path}")

        if self._text_parser.supports(file_path):
            logger.info(f"[Router] Markdown/文本文件 → TextParser：{file_path.name}")
            return self._text_parser.parse(file_path)

        if self._vision_parser.supports(file_path):
            logger.info(f"[Router] 图像文件 → VisionParser：{file_path.name}")
            return self._vision_parser.parse(file_path)

        raise ValueError(f"不支持的文件类型：{file_path.suffix}")
