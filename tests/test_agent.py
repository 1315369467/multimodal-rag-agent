"""
RAG Agent 的单元测试。
所有 LLM 和检索调用均已 mock，避免 API 依赖。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.tools import build_retrieval_tools


# ── 工具：knowledge_base_search ───────────────────────────────────────────

class TestKnowledgeBaseSearchTool:
    def _make_retriever(self, docs: list[Document]) -> MagicMock:
        retriever = MagicMock()
        retriever.retrieve.return_value = docs
        return retriever

    def _make_doc(self, content: str, source: str = "test.pdf", page: int = 1) -> Document:
        return Document(
            page_content=content,
            metadata={"source": source, "page_num": page, "block_type": "text"},
        )

    def test_returns_json_with_results(self):
        docs = [self._make_doc("The capital of France is Paris.", page=5)]
        retriever = self._make_retriever(docs)
        tools = build_retrieval_tools(retriever)

        search_tool = next(t for t in tools if t.name == "knowledge_base_search")
        result = search_tool.invoke({"query": "capital of France"})
        data = json.loads(result)

        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["content"] == "The capital of France is Paris."
        assert data["results"][0]["source"] == "test.pdf"
        assert data["results"][0]["page"] == 5

    def test_empty_results(self):
        retriever = self._make_retriever([])
        tools = build_retrieval_tools(retriever)
        search_tool = next(t for t in tools if t.name == "knowledge_base_search")
        result = search_tool.invoke({"query": "nonexistent topic"})
        data = json.loads(result)
        assert data["results"] == []

    def test_multiple_results_ranked(self):
        docs = [
            self._make_doc(f"Document {i}", page=i)
            for i in range(5)
        ]
        retriever = self._make_retriever(docs)
        tools = build_retrieval_tools(retriever)
        search_tool = next(t for t in tools if t.name == "knowledge_base_search")
        result = search_tool.invoke({"query": "document"})
        data = json.loads(result)

        ranks = [r["rank"] for r in data["results"]]
        assert ranks == list(range(1, len(ranks) + 1))


# ── 来源提取（适配 LangGraph 消息格式）──────────────────────────────────

class TestSourceExtraction:
    def test_extract_sources_from_tool_messages(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        observation = json.dumps(
            {
                "results": [
                    {"source": "doc.pdf", "page": 3, "block_type": "text", "content": "...", "rank": 1}
                ]
            }
        )
        messages = [
            HumanMessage(content="测试问题"),
            ToolMessage(content=observation, tool_call_id="call_1"),
            AIMessage(content="最终回答"),
        ]
        sources = MultimodalRAGAgent._extract_sources(messages)
        assert len(sources) == 1
        assert sources[0]["source"] == "doc.pdf"
        assert sources[0]["page"] == "3"

    def test_deduplicates_sources(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        obs = json.dumps(
            {
                "results": [
                    {"source": "doc.pdf", "page": 1, "block_type": "text", "content": "a", "rank": 1},
                    {"source": "doc.pdf", "page": 1, "block_type": "text", "content": "b", "rank": 2},
                ]
            }
        )
        messages = [
            ToolMessage(content=obs, tool_call_id="call_1"),
        ]
        sources = MultimodalRAGAgent._extract_sources(messages)
        assert len(sources) == 1  # 同一 source:page 已去重

    def test_handles_invalid_json_gracefully(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        messages = [
            ToolMessage(content="not valid json", tool_call_id="call_1"),
        ]
        sources = MultimodalRAGAgent._extract_sources(messages)
        assert sources == []

    def test_ignores_non_tool_messages(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        sources = MultimodalRAGAgent._extract_sources(messages)
        assert sources == []

    def test_multiple_tool_messages_merged(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        obs1 = json.dumps(
            {"results": [{"source": "a.pdf", "page": 1, "block_type": "text", "content": "x", "rank": 1}]}
        )
        obs2 = json.dumps(
            {"results": [{"source": "b.pdf", "page": 2, "block_type": "table", "content": "y", "rank": 1}]}
        )
        messages = [
            ToolMessage(content=obs1, tool_call_id="call_1"),
            ToolMessage(content=obs2, tool_call_id="call_2"),
        ]
        sources = MultimodalRAGAgent._extract_sources(messages)
        assert len(sources) == 2
        source_names = {s["source"] for s in sources}
        assert source_names == {"a.pdf", "b.pdf"}


# ── 历史消息格式化 ───────────────────────────────────────────────────────

class TestFormatHistory:
    def test_empty_history(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        result = MultimodalRAGAgent._format_history([])
        assert result == []

    def test_converts_roles_correctly(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = MultimodalRAGAgent._format_history(history)
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert result[0].content == "你好"
        assert result[1].content == "你好！"

    def test_ignores_unknown_roles(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        history = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "问题"},
        ]
        result = MultimodalRAGAgent._format_history(history)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
