"""
所有文档解析器的抽象基类。
每个解析器返回一个 ParsedBlock 列表——这是贯穿整条 Pipeline 的统一数据结构。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"      # 视觉模型提取的图像或图表
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class ParsedBlock:
    """
    流经 Pipeline 的最小内容单元。

    属性说明
    --------
    content:    文本内容（图表块为视觉模型生成的描述文字）。
    block_type: 该块的语义类型。
    page_num:   在源文档中的页码（0 起始）。
    metadata:   任意扩展信息（字体大小、图片路径、置信度等）。
    """
    content: str
    block_type: BlockType
    page_num: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_document(self) -> dict[str, Any]:
        """序列化为适合构造 LangChain Document 的字典格式。"""
        return {
            "page_content": self.content,
            "metadata": {
                "block_type": self.block_type.value,
                "page_num": self.page_num,
                **self.metadata,
            },
        }


class BaseDocumentParser(abc.ABC):
    """每个解析器必须实现的契约接口。"""

    @abc.abstractmethod
    def parse(self, file_path: Path) -> list[ParsedBlock]:
        """
        解析 file_path 并返回有序的 ParsedBlock 列表。
        实现类必须保持无状态，以便实例可以在多线程间共享。
        """

    def supports(self, file_path: Path) -> bool:
        """若当前解析器能处理该文件扩展名，返回 True。"""
        return file_path.suffix.lower() in self.supported_extensions

    @property
    @abc.abstractmethod
    def supported_extensions(self) -> set[str]:
        """当前解析器支持的文件扩展名集合，例如 {'.pdf'}。"""
