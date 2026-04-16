"""
纯文本 / Markdown 文件解析器
──────────────────────────────────────────────────────────────────────────────
将 .txt 和 .md 文件解析为 ParsedBlock 列表。

Markdown 解析策略
-----------------
• 标题行（# ~ ######）          → HEADER 块（携带 heading_level 元数据）
• HTML 表格（<table>...</table>）→ TABLE 块（subtype=html_table）
• Markdown 表格（| ... |）       → TABLE 块（原子块，不会被截断）
• 围栏代码块（``` / ~~~）        → TEXT 块（subtype=code_block）
• 数学公式块（$$ ... $$）        → FORMULA 块（subtype=display_math，原子块）
  └─ 支持多行 $$ 围栏格式 和 单行 $$...$$ 格式
• 独立图片行（![alt](url)）       → FIGURE 块（携带 image_url / alt_text）
  └─ 若紧随一行 "Figure N. / | / :" 标题，合并为带 caption 的 FIGURE 块
• 表格前的 "Table N. / | / :" 标题行 → 捕获为 TABLE 块的 caption 元数据
• 其余段落                      → TEXT 块（按空行分段）

这些类型信息将被 StructureAwareChunker 用于结构感知切分：
TABLE / FIGURE / FORMULA 作为原子块整体保留，不跨块截断。
"""
from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from .base_parser import BaseDocumentParser, BlockType, ParsedBlock


class TextParser(BaseDocumentParser):
    """
    解析 .txt 和 .md 文件。

    Markdown 文件会识别标题、HTML 表格、Markdown 表格、图片（带/不带标题）
    和代码块，分别映射到对应的 BlockType，供下游语义切分器使用。
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

        if is_markdown:
            blocks = self._parse_markdown(text)
        else:
            blocks = self._parse_plain_text(text)

        logger.info(f"[TextParser] {file_path.name}: {len(blocks)} blocks")
        return blocks

    # ──────────────────────────────────────────────────────────────────────
    # 正则表达式
    # ──────────────────────────────────────────────────────────────────────

    _MD_HEADING_RE        = re.compile(r"^(#{1,6})\s+(.+)$")
    _MD_IMAGE_RE          = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")
    _FENCE_RE             = re.compile(r"^(`{3,}|~{3,})")
    _HTML_TABLE_START_RE  = re.compile(r"^<table", re.IGNORECASE)
    _HTML_TABLE_END_RE    = re.compile(r"</table>", re.IGNORECASE)
    # 匹配 "Figure 1. / Figure 1 | / Figure 1:" 等图注行
    _FIGURE_CAPTION_RE    = re.compile(r"^Figure\s+\S+[\s.\|:]", re.IGNORECASE)
    # 匹配 "Table 1. / Table 1 | / Table 1:" 等表注行
    _TABLE_CAPTION_RE     = re.compile(r"^Table\s+\d+\S*[\s.\|:]", re.IGNORECASE)
    # 匹配单行 $$...$$ 数学公式（行首尾均为 $$，中间有内容）
    _INLINE_DISPLAY_MATH_RE = re.compile(r"^\$\$.+\$\$$", re.DOTALL)

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _parse_markdown(self, text: str) -> list[ParsedBlock]:
        """
        逐行扫描，按块类型分段收集：
          HEADER → 单行标题
          TABLE  → HTML <table> 或连续的 | 起始行（带可选的表注）
          FIGURE → 独立图片行（自动合并紧随其后的 Figure N 标题行）
          CODE   → 围栏代码块（整体作为 TEXT 块保留）
          TEXT   → 其余段落
        """
        blocks: list[ParsedBlock] = []
        lines = text.split("\n")
        i = 0
        text_lines: list[str] = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # ── 标题 ──────────────────────────────────────────────────────
            heading_match = self._MD_HEADING_RE.match(line)
            if heading_match:
                self._flush_text(text_lines, blocks)
                text_lines = []
                level = len(heading_match.group(1))
                blocks.append(
                    ParsedBlock(
                        content=line.rstrip(),
                        block_type=BlockType.HEADER,
                        page_num=0,
                        metadata={"heading_level": level},
                    )
                )
                i += 1
                continue

            # ── 数学公式块（多行 $$ ... $$ 围栏） ────────────────────────
            if stripped == "$$":
                self._flush_text(text_lines, blocks)
                text_lines = []
                math_lines = [line]
                i += 1
                while i < len(lines):
                    math_lines.append(lines[i])
                    if lines[i].strip() == "$$":
                        i += 1
                        break
                    i += 1
                content = "\n".join(math_lines).strip()
                if content:
                    blocks.append(
                        ParsedBlock(
                            content=content,
                            block_type=BlockType.FORMULA,
                            page_num=0,
                            metadata={"subtype": "display_math"},
                        )
                    )
                continue

            # ── 数学公式块（单行 $$...$$ 格式） ──────────────────────────
            if self._INLINE_DISPLAY_MATH_RE.match(stripped):
                self._flush_text(text_lines, blocks)
                text_lines = []
                blocks.append(
                    ParsedBlock(
                        content=stripped,
                        block_type=BlockType.FORMULA,
                        page_num=0,
                        metadata={"subtype": "display_math"},
                    )
                )
                i += 1
                continue

            # ── 围栏代码块 ────────────────────────────────────────────────
            fence_match = self._FENCE_RE.match(stripped)
            if fence_match:
                self._flush_text(text_lines, blocks)
                text_lines = []
                fence_str = fence_match.group(1)
                code_lines = [line]
                i += 1
                while i < len(lines):
                    code_lines.append(lines[i])
                    if lines[i].strip().startswith(fence_str) and lines[i].strip() != line.strip():
                        i += 1
                        break
                    i += 1
                content = "\n".join(code_lines).strip()
                if content:
                    blocks.append(
                        ParsedBlock(
                            content=content,
                            block_type=BlockType.TEXT,
                            page_num=0,
                            metadata={"subtype": "code_block"},
                        )
                    )
                continue

            # ── HTML 表格 ─────────────────────────────────────────────────
            if self._HTML_TABLE_START_RE.match(stripped):
                self._flush_text(text_lines, blocks)
                text_lines = []
                # 尝试从上一个 TEXT 块中提取表注
                caption = self._pop_caption(blocks, self._TABLE_CAPTION_RE)
                table_lines = [line]
                if not self._HTML_TABLE_END_RE.search(line):
                    i += 1
                    while i < len(lines):
                        table_lines.append(lines[i])
                        if self._HTML_TABLE_END_RE.search(lines[i]):
                            i += 1
                            break
                        i += 1
                else:
                    i += 1
                table_content = "\n".join(table_lines).strip()
                if table_content:
                    # 将表注并入内容开头，方便向量检索命中
                    content = f"{caption}\n{table_content}".strip() if caption else table_content
                    blocks.append(
                        ParsedBlock(
                            content=content,
                            block_type=BlockType.TABLE,
                            page_num=0,
                            metadata={"subtype": "html_table", "caption": caption},
                        )
                    )
                continue

            # ── Markdown 表格 ─────────────────────────────────────────────
            if stripped.startswith("|"):
                self._flush_text(text_lines, blocks)
                text_lines = []
                caption = self._pop_caption(blocks, self._TABLE_CAPTION_RE)
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                table_content = "\n".join(table_lines).strip()
                if table_content:
                    content = f"{caption}\n{table_content}".strip() if caption else table_content
                    blocks.append(
                        ParsedBlock(
                            content=content,
                            block_type=BlockType.TABLE,
                            page_num=0,
                            metadata={"caption": caption} if caption else {},
                        )
                    )
                continue

            # ── 独立图片行 ────────────────────────────────────────────────
            img_match = self._MD_IMAGE_RE.match(stripped)
            if img_match:
                self._flush_text(text_lines, blocks)
                text_lines = []
                alt = img_match.group(1)
                url = img_match.group(2)
                i += 1
                # 若下一行紧接着是 "Figure N ..." 标题，合并为图注内容
                caption = ""
                if i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped and self._FIGURE_CAPTION_RE.match(next_stripped):
                        caption = next_stripped
                        i += 1
                content = caption if caption else (f"[图片: {alt}]" if alt else "[图片]")
                blocks.append(
                    ParsedBlock(
                        content=content,
                        block_type=BlockType.FIGURE,
                        page_num=0,
                        metadata={"image_url": url, "alt_text": alt, "caption": caption},
                    )
                )
                continue

            # ── 普通文本行 ────────────────────────────────────────────────
            text_lines.append(line)
            i += 1

        self._flush_text(text_lines, blocks)
        return blocks

    def _parse_plain_text(self, text: str) -> list[ParsedBlock]:
        """按空行分段，每段生成一个 TEXT 块。"""
        blocks: list[ParsedBlock] = []
        for para in re.split(r"\n\s*\n", text):
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
    def _pop_caption(blocks: list[ParsedBlock], pattern: re.Pattern) -> str:
        """
        若 blocks 末尾是匹配 pattern 的单行 TEXT 块（表注 / 图注），
        将其从列表中弹出并返回其文本，否则返回空字符串。
        """
        if blocks and blocks[-1].block_type == BlockType.TEXT:
            content = blocks[-1].content
            # 仅对单行（无换行）的文本块做匹配，避免误吞多段落文本
            if "\n" not in content and pattern.match(content):
                blocks.pop()
                return content
        return ""

    @staticmethod
    def _flush_text(lines: list[str], blocks: list[ParsedBlock]) -> None:
        """将累积的文本行按空行分段，追加到 blocks 列表。"""
        text = "\n".join(lines)
        for para in re.split(r"\n\s*\n", text):
            content = para.strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content,
                        block_type=BlockType.TEXT,
                        page_num=0,
                    )
                )
