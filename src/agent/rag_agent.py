"""
多模态 RAG Agent
──────────────────────────────────────────────────────────────────────────────
基于 LangGraph ReAct 框架，通过检索知识库后用 Qwen3.5（DashScope）生成答案。

架构示意
────────
  用户问题
       │
       ▼
  ┌────────────────────────────────────────┐
  │          ReAct Agent 循环              │
  │  ┌──────────┐    ┌───────────────────┐ │
  │  │  Qwen3.5 │◄──►│   工具执行器      │ │
  │  │   LLM    │    │ (knowledge_base   │ │
  │  └──────────┘    │  _search)         │ │
  │                  └─────────┬─────────┘ │
  └────────────────────────────┼───────────┘
                               │
                    ┌──────────▼──────────┐
                    │   HybridRetriever   │
                    │   稠密 + 稀疏检索   │
                    │   + Reranker 精排   │
                    └─────────────────────┘

Agent 对每次查询无状态；对话历史由调用方传入，
因此该类可安全地在并发请求间共享。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from config.settings import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from .tools import build_retrieval_tools

settings = get_settings()

_SYSTEM_PROMPT = """\
你是一个专业的企业知识库智能助手，具备多模态文档理解能力。你的职责是：
1. 根据用户问题，精确检索知识库中的相关内容（含文本、表格、图表等）。
2. 基于检索到的上下文，给出准确、结构化、有据可查的回答。
3. 如果知识库中没有相关信息，明确告知用户，不要编造内容。
4. 引用来源时注明文档名称和页码，增强可信度。

检索策略：
- 先用 knowledge_base_search 检索通用答案。
- 如用户指定了文件，改用 knowledge_base_search_with_filter。
- 必要时可多次检索以获取不同角度的信息。

回答格式：
- 使用 Markdown 格式，结构清晰。
- 对于数据类问题，优先使用表格。
- 数学公式使用 LaTeX。
- 在回答末尾注明参考来源。
"""


class MultimodalRAGAgent:
    """
    封装了面向多模态 RAG 场景的 LangGraph ReAct Agent。

    参数
    ----
    retriever : 已初始化的 HybridRetriever（BM25 索引须已构建）。
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            request_timeout=settings.llm_request_timeout,
        )
        self._tools = build_retrieval_tools(retriever)
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=_SYSTEM_PROMPT,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        使用 RAG 流水线回答问题。

        参数
        ----
        question     : 用户的自然语言问题。
        chat_history : 可选的历史对话轮次
                       [{"role": "user"|"assistant", "content": "..."}]。

        返回
        ----
        包含 answer、sources、intermediate_steps 三个键的字典。
        """
        messages = self._format_history(chat_history or [])
        messages.append(HumanMessage(content=question))

        logger.info(f"[RAGAgent] 问题：{question[:120]}")

        result = self._agent.invoke({"messages": messages})

        # 提取最终回答（最后一条 AI 消息）
        output_messages = result.get("messages", [])
        answer = ""
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        # 从 ToolMessage 中提取来源引用
        sources = self._extract_sources(output_messages)

        logger.info(f"[RAGAgent] 答案：{len(answer)} 字 | 来源：{sources}")
        return {
            "answer": answer,
            "sources": sources,
            "intermediate_steps": [],
        }

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_history(
        history: list[dict[str, str]],
    ) -> list[HumanMessage | AIMessage]:
        """将字典格式的历史轮次转换为 LangChain 消息对象。"""
        messages: list[HumanMessage | AIMessage] = []
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def _extract_sources(messages: list) -> list[dict[str, str]]:
        """从消息列表中的 ToolMessage 提取被引用的来源信息。"""
        seen: set[str] = set()
        sources: list[dict[str, str]] = []

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else ""
            if not content:
                continue
            try:
                data = json.loads(content)
                for result in data.get("results", []):
                    key = f"{result.get('source')}:{result.get('page')}"
                    if key not in seen:
                        seen.add(key)
                        sources.append(
                            {
                                "source": result.get("source", "unknown"),
                                "page": str(result.get("page", "?")),
                                "block_type": result.get("block_type", "text"),
                            }
                        )
            except (json.JSONDecodeError, AttributeError):
                pass

        return sources
