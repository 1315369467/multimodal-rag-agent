"""
RAG Agent 的 LangChain 工具定义
──────────────────────────────────────────────────────────────────────────────
工具将检索子系统暴露给 LangChain Agent 循环。
Agent 在生成答案前可调用这些工具获取上下文。
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from loguru import logger

if TYPE_CHECKING:
    from src.retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# 工厂函数——在启动时绑定检索器，返回可用工具列表
# ---------------------------------------------------------------------------

def build_retrieval_tools(retriever: "HybridRetriever") -> list:
    """
    返回绑定到指定 HybridRetriever 的 LangChain 工具列表。

    使用工厂模式（而非模块级全局变量）的原因：
    保持工具的可测试性，避免导入时的副作用。
    """

    @tool
    def knowledge_base_search(query: str) -> str:
        """
        在多模态知识库中检索与查询相关的信息。

        当需要从已上传文档中获取事实性信息时使用此工具。
        返回包含来源元数据的相关段落 JSON 列表。

        参数
        ----
        query : 自然语言问题或搜索短语。
        """
        logger.info(f"[Tool:knowledge_base_search] query='{query[:80]}'")
        docs = retriever.retrieve(query)

        if not docs:
            return json.dumps({"results": [], "message": "未找到相关内容。"})

        results = []
        for i, doc in enumerate(docs, start=1):
            results.append(
                {
                    "rank": i,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page_num", "?"),
                    "block_type": doc.metadata.get("block_type", "text"),
                    "header_context": doc.metadata.get("header_context", ""),
                }
            )

        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    @tool
    def knowledge_base_search_with_filter(query: str, source_file: str) -> str:
        """
        在知识库中检索，结果限定在指定的源文档范围内。

        当用户询问特定文件或文档时使用此工具。

        参数
        ----
        query       : 自然语言问题。
        source_file : 限定检索的文件名（例如 'annual_report.pdf'）。
        """
        logger.info(
            f"[Tool:filtered_search] query='{query[:60]}' filter='{source_file}'"
        )
        docs = retriever.dense.retrieve(
            query, filter={"source": {"$eq": source_file}}
        )

        if not docs:
            return json.dumps(
                {
                    "results": [],
                    "message": f"在 '{source_file}' 中未找到相关内容。",
                }
            )

        results = [
            {
                "rank": i,
                "content": doc.page_content,
                "source": doc.metadata.get("source", source_file),
                "page": doc.metadata.get("page_num", "?"),
            }
            for i, (doc, _) in enumerate(docs[:5], start=1)
        ]
        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    return [knowledge_base_search, knowledge_base_search_with_filter]
