"""
纯文本 / Markdown 文件解析器
──────────────────────────────────────────────────────────────────────────────
将 .txt 和 .md 文件解析为 ParsedBlock 列表。

Markdown 文件会识别标题行（# 开头）并标记为 HEADER 类型，
其余内容按段落拆分为 TEXT 块。
"""
from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from .base_parser import BaseDocumentParser, BlockType, ParsedBlock


class TextParser(BaseDocumentParser):
    """
    解析 .txt 和 .md 文件。

    • Markdown 标题（# ~ ######）→ HEADER 块
    • 其余段落 → TEXT 块
    • 按空行分段，保持段落完整性
    """

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".markdown"}

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def parse(self, file_path: Path) -> list[ParsedBlock]:
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        is_markdown = file_path.suffix.lower() in {".md", ".markdown"}

        blocks: list[ParsedBlock] = []

        if is_markdown:
            blocks = self._parse_markdown(text)
        else:
            blocks = self._parse_plain_text(text)

        logger.info(
            f"[TextParser] {file_path.name}: {len(blocks)} blocks"
        )
        return blocks

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    _MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

    def _parse_markdown(self, text: str) -> list[ParsedBlock]:
        """按 Markdown 结构拆分：标题行 → HEADER，连续正文 → TEXT。"""
        blocks: list[ParsedBlock] = []
        current_lines: list[str] = []

        for line in text.split("\n"):
            heading_match = self._MD_HEADING_RE.match(line)
            if heading_match:
                # 先把之前积累的正文段落输出
                self._flush_text(current_lines, blocks)
                current_lines = []
                # 标题块
                level = len(heading_match.group(1))
                blocks.append(
                    ParsedBlock(
                        content=heading_match.group(2).strip(),
                        block_type=BlockType.HEADER,
                        page_num=0,
                        metadata={"heading_level": level},
                    )
                )
            else:
                current_lines.append(line)

        # 输出最后一段正文
        self._flush_text(current_lines, blocks)
        return blocks

    def _parse_plain_text(self, text: str) -> list[ParsedBlock]:
        """按空行分段，每段生成一个 TEXT 块。"""
        blocks: list[ParsedBlock] = []
        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            content = para.strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content,
                        block_type=BlockType.TEXT,
                        page_num=0,
                    )
                )

        return blocks

    @staticmethod
    def _flush_text(lines: list[str], blocks: list[ParsedBlock]) -> None:
        """将累积的文本行按空行分段，追加到 blocks 列表。"""
        text = "\n".join(lines)
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            content = para.strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content,
                        block_type=BlockType.TEXT,
                        page_num=0,
                    )
                )
