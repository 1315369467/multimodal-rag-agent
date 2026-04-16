"""
切分器模块
──────────────────────────────────────────────────────────────────────────────
提供六种切分策略，统一入口为 chunk_blocks / chunk_text。

策略说明
────────
fixed     : 按固定字符数切分（1 token ≈ 2 中文字符），无重叠。
sentence  : 按中文/英文句子边界切分，每 chunk 至多 5 句。
paragraph : 按连续空行切分，短段自动合并。
markdown  : 按 Markdown 标题（#/##/###）切分。
recursive : LangChain RecursiveCharacterTextSplitter，优先段落→句子→字符。
semantic  : 结构感知切分（StructureAwareChunker），保留 ParsedBlock 结构信息。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from .base_parser import BlockType, ParsedBlock

settings = get_settings()

# ── 公开常量 ──────────────────────────────────────────────────────────────

SUPPORTED_STRATEGIES = {"fixed", "sentence", "paragraph", "markdown", "recursive", "semantic"}

STRATEGY_LABEL = {
    "fixed":     "Fixed-size",
    "sentence":  "Sentence",
    "paragraph": "Paragraph",
    "markdown":  "Markdown-header",
    "recursive": "Recursive",
    "semantic":  "Semantic",
}

# ── 简单词数代理估算 token 数 ──────────────────────────────────────────────

_WORDS_PER_TOKEN = 0.75


def _word_count(text: str) -> int:
    return len(text.split())


def _token_estimate(text: str) -> int:
    return int(_word_count(text) / _WORDS_PER_TOKEN)


# ── 内部工具 ──────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "page_num": 0})


# ══════════════════════════════════════════════════════════════════════════════
# 文本级切分策略
# ══════════════════════════════════════════════════════════════════════════════

def chunk_fixed(text: str, source: str, chunk_size: int = 256) -> list[Document]:
    """按固定字符数切分（无重叠）。chunk_size 单位：近似 token 数。"""
    char_size = chunk_size * 2  # 粗略：1 token ≈ 2 中文字符
    docs = []
    start = 0
    while start < len(text):
        segment = text[start: start + char_size].strip()
        if segment:
            docs.append(_make_doc(segment, source))
        start += char_size
    return docs


def chunk_sentence(text: str, source: str, max_sentences: int = 5) -> list[Document]:
    """
    按中文句子边界（。！？… 及英文 .!?）切分，
    每个 chunk 包含至多 max_sentences 句。
    """
    sentences = re.split(r"(?<=[。！？…\.!?])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    docs = []
    for i in range(0, len(sentences), max_sentences):
        segment = "".join(sentences[i: i + max_sentences]).strip()
        if segment:
            docs.append(_make_doc(segment, source))
    return docs


def chunk_paragraph(text: str, source: str, min_len: int = 20) -> list[Document]:
    """按段落（连续空行）切分，段落太短时与下一段合并。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs = []
    buffer = ""
    for para in paragraphs:
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= min_len:
            docs.append(_make_doc(buffer, source))
            buffer = ""
    if buffer:
        docs.append(_make_doc(buffer, source))
    return docs


def chunk_markdown(text: str, source: str, min_len: int = 20) -> list[Document]:
    """按 Markdown 标题（#/##/###）切分。"""
    sections = re.split(r"\n(?=#{1,3}\s)", text)
    docs = []
    for section in sections:
        section = section.strip()
        if section and len(section) >= min_len:
            docs.append(_make_doc(section, source))
    return docs


def chunk_recursive(
    text: str,
    source: str,
    chunk_size: int = 256,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    使用 LangChain RecursiveCharacterTextSplitter，
    优先在段落/句子/字符边界处切分。
    chunk_overlap 默认为 chunk_size // 4。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if chunk_overlap is None:
        chunk_overlap = chunk_size // 4
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 2,
        chunk_overlap=chunk_overlap * 2,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [_make_doc(c.strip(), source) for c in chunks if c.strip()]


# ── 策略分发表 ────────────────────────────────────────────────────────────

STRATEGY_FN: dict[str, Any] = {
    "fixed":     chunk_fixed,
    "sentence":  chunk_sentence,
    "paragraph": chunk_paragraph,
    "markdown":  chunk_markdown,
    "recursive": chunk_recursive,
}


# ── 高层文本入口 ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    strategy: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """对原始文本应用指定策略，返回 Document 列表。"""
    if strategy not in STRATEGY_FN:
        raise ValueError(f"未知切分策略：{strategy}，可选：{sorted(STRATEGY_FN)}")
    fn = STRATEGY_FN[strategy]
    kwargs: dict = {}
    if strategy in ("fixed", "recursive"):
        kwargs["chunk_size"] = chunk_size
    if strategy == "recursive":
        kwargs["chunk_overlap"] = chunk_overlap
    return fn(text, source, **kwargs)  # type: ignore[operator]


# ══════════════════════════════════════════════════════════════════════════════
# 版面感知切分（Semantic）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ChunkBuffer:
    parts: list[str]
    header_context: str = ""
    page_num: int = 0
    block_types: list[str] | None = None

    def __post_init__(self):
        self.block_types = self.block_types or []

    @property
    def text(self) -> str:
        pieces = []
        if self.header_context:
            pieces.append(self.header_context)
        pieces.extend(self.parts)
        return "\n\n".join(p for p in pieces if p.strip())

    @property
    def estimated_tokens(self) -> int:
        return _token_estimate(self.text)

    def is_empty(self) -> bool:
        return not any(p.strip() for p in self.parts)


class StructureAwareChunker:
    """
    将 ParsedBlock 转换为结构感知切分的 LangChain Document。

    参数
    ----
    chunk_size    : 目标 chunk 大小（近似 token 数）。
    chunk_overlap : 相邻 chunk 之间的重叠量（近似 token 数）。
    source_name   : 原始文档名称，写入每个 Document 的元数据。
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        source_name: str = "",
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.source_name = source_name

    def chunk(self, blocks: list[ParsedBlock]) -> list[Document]:
        """主入口。返回扁平的 LangChain Document 列表。"""
        documents: list[Document] = []
        buffer = _ChunkBuffer(parts=[], page_num=blocks[0].page_num if blocks else 0)
        current_header = ""

        for block in blocks:
            block_type = block.block_type

            # ── 标题块：刷新缓冲区，存储上下文 ──────────────────────────
            if block_type == BlockType.HEADER:
                if not buffer.is_empty():
                    documents.extend(self._flush(buffer))
                current_header = block.content.strip()
                buffer = _ChunkBuffer(
                    parts=[], header_context=current_header, page_num=block.page_num
                )
                continue

            # ── 原子块：TABLE / FIGURE / FORMULA ─────────────────────────
            if block_type in (BlockType.TABLE, BlockType.FIGURE, BlockType.FORMULA):
                if not buffer.is_empty():
                    documents.extend(self._flush(buffer))
                    buffer = _ChunkBuffer(
                        parts=[], header_context=current_header, page_num=block.page_num
                    )
                meta = self._build_meta(block, current_header)
                doc = Document(
                    page_content=f"{current_header}\n\n{block.content}".strip()
                    if current_header
                    else block.content,
                    metadata=meta,
                )
                documents.append(doc)
                continue

            # ── 普通文本块 ─────────────────────────────────────────────────
            if block.page_num != buffer.page_num and not buffer.is_empty():
                documents.extend(self._flush(buffer))
                buffer = _ChunkBuffer(
                    parts=[], header_context=current_header, page_num=block.page_num
                )

            curr_font = block.metadata.get("avg_font_size", 12)
            _ = curr_font  # 扩展实现中使用；此处抑制 lint 警告

            candidate = block.content.strip()
            if not candidate:
                continue

            if (
                buffer.estimated_tokens + _token_estimate(candidate)
                > self.chunk_size
                and not buffer.is_empty()
            ):
                documents.extend(self._flush(buffer))
                overlap_text = self._extract_overlap(buffer)
                buffer = _ChunkBuffer(
                    parts=[overlap_text] if overlap_text else [],
                    header_context=current_header,
                    page_num=block.page_num,
                )

            buffer.parts.append(candidate)
            buffer.page_num = block.page_num
            buffer.block_types.append(block_type.value)

        if not buffer.is_empty():
            documents.extend(self._flush(buffer))

        logger.info(
            f"[Chunker] {self.source_name}: "
            f"{len(blocks)} blocks → {len(documents)} chunks"
        )
        return documents

    def _flush(self, buffer: _ChunkBuffer) -> list[Document]:
        text = buffer.text.strip()
        if not text:
            return []
        meta: dict[str, Any] = {
            "source": self.source_name,
            "page_num": buffer.page_num,
            "block_types": list(set(buffer.block_types or [])),
            "header_context": buffer.header_context,
            "chunk_tokens": _token_estimate(text),
        }
        return [Document(page_content=text, metadata=meta)]

    def _extract_overlap(self, buffer: _ChunkBuffer) -> str:
        words = buffer.text.split()
        n_words = int(self.chunk_overlap / _WORDS_PER_TOKEN)
        return " ".join(words[-n_words:]) if len(words) > n_words else " ".join(words)

    def _build_meta(self, block: ParsedBlock, header_context: str) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "page_num": block.page_num,
            "block_type": block.block_type.value,
            "header_context": header_context,
            **block.metadata,
        }


# ── 高层 Block 入口 ───────────────────────────────────────────────────────

def chunk_blocks(
    blocks: list[ParsedBlock],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    source_name: str,
) -> list[Document]:
    """
    对 ParsedBlock 列表应用指定策略，返回 Document 列表。
    strategy == "semantic" 时使用 StructureAwareChunker 保留版面结构；
    其余策略将块内容合并为文本后调用 chunk_text。
    """
    if strategy == "semantic":
        return StructureAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_name=source_name,
        ).chunk(blocks)

    parts = []
    for b in blocks:
        if not b.content.strip():
            continue
        if b.block_type == BlockType.HEADER:
            level = b.metadata.get("heading_level", 2)
            parts.append("#" * level + " " + b.content.strip())
        else:
            parts.append(b.content.strip())
    text = "\n\n".join(parts)
    return chunk_text(text, source_name, strategy, chunk_size, chunk_overlap)
