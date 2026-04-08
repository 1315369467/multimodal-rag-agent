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


# ── 工具：query_rewrite ───────────────────────────────────────────────────

class TestQueryRewriteTool:
    def _make_retriever(self) -> MagicMock:
        return MagicMock()

    def _make_llm(self, response_content: str) -> MagicMock:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=response_content)
        return llm

    def test_returns_rewritten_queries(self):
        llm = self._make_llm('{"queries": ["子查询A", "子查询B", "子查询C"]}')
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)

        rewrite_tool = next(t for t in tools if t.name == "query_rewrite")
        result = rewrite_tool.invoke({"query": "这家公司的年利润是多少？"})
        data = json.loads(result)

        assert "original_query" in data
        assert "rewritten_queries" in data
        assert data["original_query"] == "这家公司的年利润是多少？"
        assert data["rewritten_queries"] == ["子查询A", "子查询B", "子查询C"]

    def test_falls_back_to_original_on_llm_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)

        rewrite_tool = next(t for t in tools if t.name == "query_rewrite")
        result = rewrite_tool.invoke({"query": "原始问题"})
        data = json.loads(result)

        assert data["rewritten_queries"] == ["原始问题"]

    def test_falls_back_on_malformed_json(self):
        llm = self._make_llm("这不是合法的JSON")
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)

        rewrite_tool = next(t for t in tools if t.name == "query_rewrite")
        result = rewrite_tool.invoke({"query": "问题X"})
        data = json.loads(result)

        assert data["rewritten_queries"] == ["问题X"]

    def test_strips_markdown_code_block(self):
        llm = self._make_llm('```json\n{"queries": ["q1", "q2", "q3"]}\n```')
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)

        rewrite_tool = next(t for t in tools if t.name == "query_rewrite")
        result = rewrite_tool.invoke({"query": "问题Y"})
        data = json.loads(result)

        assert data["rewritten_queries"] == ["q1", "q2", "q3"]

    def test_not_registered_without_llm(self):
        tools = build_retrieval_tools(self._make_retriever(), llm=None)
        names = [t.name for t in tools]
        assert "query_rewrite" not in names

    def test_registered_with_llm(self):
        llm = self._make_llm('{"queries": ["a"]}')
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)
        names = [t.name for t in tools]
        assert "query_rewrite" in names

    def test_falls_back_on_empty_queries_list(self):
        llm = self._make_llm('{"queries": []}')
        tools = build_retrieval_tools(self._make_retriever(), llm=llm)

        rewrite_tool = next(t for t in tools if t.name == "query_rewrite")
        result = rewrite_tool.invoke({"query": "空列表问题"})
        data = json.loads(result)

        assert data["rewritten_queries"] == ["空列表问题"]


# ── 工具：multi_round_search ──────────────────────────────────────────────

class TestMultiRoundSearchTool:
    def _make_doc(self, content: str, source: str = "a.pdf", page: int = 1) -> "Document":
        from langchain_core.documents import Document
        return Document(
            page_content=content,
            metadata={"source": source, "page_num": page, "block_type": "text"},
        )

    def _make_retriever(self, docs_per_query: dict[str, list]) -> MagicMock:
        retriever = MagicMock()
        retriever.retrieve.side_effect = lambda q: docs_per_query.get(q, [])
        return retriever

    def _get_tool(self, retriever):
        tools = build_retrieval_tools(retriever)
        return next(t for t in tools if t.name == "multi_round_search")

    def test_merges_results_from_multiple_queries(self):
        docs = {
            "查询1": [self._make_doc("内容A", page=1)],
            "查询2": [self._make_doc("内容B", page=2)],
        }
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": json.dumps(["查询1", "查询2"])})
        data = json.loads(result)

        assert data["total_queries"] == 2
        contents = [r["content"] for r in data["results"]]
        assert "内容A" in contents
        assert "内容B" in contents

    def test_deduplicates_identical_content(self):
        same_doc = self._make_doc("重复内容" * 20)  # 确保前120字符相同
        docs = {
            "查询1": [same_doc],
            "查询2": [same_doc],
        }
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": json.dumps(["查询1", "查询2"])})
        data = json.loads(result)

        assert len(data["results"]) == 1

    def test_includes_matched_query_field(self):
        docs = {"查询X": [self._make_doc("内容X")]}
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": json.dumps(["查询X"])})
        data = json.loads(result)

        assert data["results"][0]["matched_query"] == "查询X"

    def test_empty_results_when_no_docs_found(self):
        tool = self._get_tool(self._make_retriever({}))
        result = tool.invoke({"queries": json.dumps(["找不到的查询"])})
        data = json.loads(result)

        assert data["results"] == []
        assert "message" in data

    def test_fallback_on_non_json_input(self):
        docs = {"普通字符串": [self._make_doc("内容C")]}
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": "普通字符串"})
        data = json.loads(result)

        assert len(data["results"]) == 1
        assert data["results"][0]["content"] == "内容C"

    def test_ranks_are_sequential(self):
        docs = {
            "q1": [self._make_doc("A"), self._make_doc("B")],
            "q2": [self._make_doc("C")],
        }
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": json.dumps(["q1", "q2"])})
        data = json.loads(result)

        ranks = [r["rank"] for r in data["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_skips_empty_queries(self):
        docs = {"有效查询": [self._make_doc("内容D")]}
        tool = self._get_tool(self._make_retriever(docs))
        result = tool.invoke({"queries": json.dumps(["有效查询", "", "  "])})
        data = json.loads(result)

        assert len(data["results"]) == 1
