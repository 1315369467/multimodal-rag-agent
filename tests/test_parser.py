"""
文档解析子系统的单元测试。
使用轻量级 mock，避免真实 API 调用和大文件 I/O。
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.document_parser.base_parser import BlockType, BoundingBox, ParsedBlock
from src.document_parser.chunker import StructureAwareChunker
from src.document_parser.text_parser import TextParser


# ── ParsedBlock / BoundingBox ─────────────────────────────────────────────

class TestBoundingBox:
    def test_area(self):
        bbox = BoundingBox(0.1, 0.1, 0.5, 0.6, page=0)
        assert abs(bbox.area - 0.2) < 1e-6

    def test_zero_area(self):
        bbox = BoundingBox(0.5, 0.5, 0.5, 0.5)
        assert bbox.area == 0.0


class TestParsedBlock:
    def test_to_langchain_document(self):
        block = ParsedBlock(
            content="Hello world",
            block_type=BlockType.TEXT,
            bbox=BoundingBox(0, 0, 1, 1),
            page_num=2,
            metadata={"avg_font_size": 12.0},
        )
        doc = block.to_langchain_document()
        assert doc["page_content"] == "Hello world"
        assert doc["metadata"]["page_num"] == 2
        assert doc["metadata"]["block_type"] == "text"
        assert doc["metadata"]["avg_font_size"] == 12.0

    def test_to_langchain_document_no_bbox(self):
        block = ParsedBlock(content="No bbox", block_type=BlockType.HEADER)
        doc = block.to_langchain_document()
        assert doc["metadata"]["bbox"] is None


# ── TextParser ────────────────────────────────────────────────────────────

class TestTextParser:
    def _write_temp_file(self, content: str, suffix: str) -> Path:
        """写入临时文件并返回路径。"""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return Path(f.name)

    def test_supported_extensions(self):
        parser = TextParser()
        assert parser.supports(Path("test.txt"))
        assert parser.supports(Path("test.md"))
        assert parser.supports(Path("test.markdown"))
        assert not parser.supports(Path("test.pdf"))

    def test_parse_plain_text_paragraphs(self):
        content = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        path = self._write_temp_file(content, ".txt")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            assert len(blocks) == 3
            assert all(b.block_type == BlockType.TEXT for b in blocks)
            assert blocks[0].content == "第一段内容。"
            assert blocks[2].content == "第三段内容。"
        finally:
            path.unlink()

    def test_parse_plain_text_empty(self):
        path = self._write_temp_file("", ".txt")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            assert blocks == []
        finally:
            path.unlink()

    def test_parse_markdown_headings(self):
        content = "# 标题一\n\n正文内容。\n\n## 标题二\n\n更多内容。"
        path = self._write_temp_file(content, ".md")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            headers = [b for b in blocks if b.block_type == BlockType.HEADER]
            texts = [b for b in blocks if b.block_type == BlockType.TEXT]
            assert len(headers) == 2
            assert headers[0].content == "标题一"
            assert headers[0].metadata["heading_level"] == 1
            assert headers[1].content == "标题二"
            assert headers[1].metadata["heading_level"] == 2
            assert len(texts) >= 2
        finally:
            path.unlink()

    def test_parse_markdown_heading_levels(self):
        content = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        path = self._write_temp_file(content, ".md")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            headers = [b for b in blocks if b.block_type == BlockType.HEADER]
            assert len(headers) == 6
            levels = [b.metadata["heading_level"] for b in headers]
            assert levels == [1, 2, 3, 4, 5, 6]
        finally:
            path.unlink()

    def test_parse_markdown_no_headings(self):
        """无标题的 Markdown 应全部解析为 TEXT 块。"""
        content = "普通段落一。\n\n普通段落二。"
        path = self._write_temp_file(content, ".md")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            assert all(b.block_type == BlockType.TEXT for b in blocks)
            assert len(blocks) == 2
        finally:
            path.unlink()

    def test_page_num_always_zero(self):
        """纯文本文件没有页码概念，page_num 始终为 0。"""
        content = "一段文字。"
        path = self._write_temp_file(content, ".txt")
        try:
            parser = TextParser()
            blocks = parser.parse(path)
            assert all(b.page_num == 0 for b in blocks)
        finally:
            path.unlink()


# ── StructureAwareChunker ───────────────────────────────────────────────────────

class TestStructureAwareChunker:
    def _make_text_block(self, content: str, page: int = 0) -> ParsedBlock:
        return ParsedBlock(
            content=content,
            block_type=BlockType.TEXT,
            bbox=BoundingBox(0, 0, 1, 0.1, page),
            page_num=page,
        )

    def _make_header_block(self, content: str, page: int = 0) -> ParsedBlock:
        return ParsedBlock(
            content=content,
            block_type=BlockType.HEADER,
            page_num=page,
            metadata={"avg_font_size": 16.0},
        )

    def _make_table_block(self, content: str, page: int = 0) -> ParsedBlock:
        return ParsedBlock(
            content=content,
            block_type=BlockType.TABLE,
            page_num=page,
        )

    def test_empty_input(self):
        chunker = StructureAwareChunker(chunk_size=512, source_name="test.pdf")
        assert chunker.chunk([]) == []

    def test_single_text_block(self):
        chunker = StructureAwareChunker(chunk_size=512, source_name="test.pdf")
        blocks = [self._make_text_block("Short paragraph.")]
        docs = chunker.chunk(blocks)
        assert len(docs) == 1
        assert docs[0].page_content == "Short paragraph."
        assert docs[0].metadata["source"] == "test.pdf"

    def test_table_is_atomic(self):
        chunker = StructureAwareChunker(chunk_size=512, source_name="test.pdf")
        blocks = [
            self._make_text_block("Before table."),
            self._make_table_block("col1 | col2\nval1 | val2"),
            self._make_text_block("After table."),
        ]
        docs = chunker.chunk(blocks)
        # 表格必须单独成为一个 Document
        table_docs = [d for d in docs if "table" in d.metadata.get("block_types", [])
                      or d.metadata.get("block_type") == "table"]
        assert len(table_docs) >= 1
        assert any("col1" in d.page_content for d in docs)

    def test_header_becomes_context(self):
        chunker = StructureAwareChunker(chunk_size=512, source_name="test.pdf")
        blocks = [
            self._make_header_block("Chapter 1: Introduction"),
            self._make_text_block("This chapter covers the basics."),
        ]
        docs = chunker.chunk(blocks)
        assert len(docs) >= 1
        # 标题上下文应出现在后续 chunk 的内容或元数据中
        combined_text = " ".join(d.page_content for d in docs)
        assert "Introduction" in combined_text

    def test_chunk_size_respected(self):
        chunker = StructureAwareChunker(chunk_size=10, source_name="test.pdf")
        # 每个块约 20 token，应产生至少 2 个 chunk
        blocks = [
            self._make_text_block(
                "This is a fairly long paragraph with many words that exceed the chunk size limit."
            )
            for _ in range(3)
        ]
        docs = chunker.chunk(blocks)
        assert len(docs) >= 2

    def test_metadata_source_set(self):
        chunker = StructureAwareChunker(source_name="my_doc.pdf")
        blocks = [self._make_text_block("content")]
        docs = chunker.chunk(blocks)
        for doc in docs:
            assert doc.metadata["source"] == "my_doc.pdf"
