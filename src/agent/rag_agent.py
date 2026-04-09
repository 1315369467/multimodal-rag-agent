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
from .skills import build_skill_tools

settings = get_settings()


def _summarize_tool_result(tool_name: str, raw: str) -> str:
    """将工具返回的 JSON 转为一句人可读的摘要，用于前端展示。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:120] if raw else ""

    results = data.get("results", [])
    message = data.get("message", "")

    if tool_name == "query_rewrite":
        queries = data.get("rewritten_queries", [])
        return f"生成 {len(queries)} 个子查询：" + "、".join(f'"{q}"' for q in queries[:3])

    if tool_name in ("knowledge_base_search", "multi_round_search",
                     "knowledge_base_search_with_filter"):
        if not results:
            return message or "未找到相关内容"
        total = data.get("total_queries")
        prefix = f"（共 {total} 轮检索）" if total else ""
        snippets = "；".join(
            r.get("content", "")[:40] for r in results[:2]
        )
        return f"找到 {len(results)} 个片段{prefix}：{snippets}…"

    # 通用兜底
    if results:
        return f"返回 {len(results)} 条结果"
    return raw[:120]


_SYSTEM_PROMPT = """\
你是一个专业的企业知识库智能助手，具备多模态文档理解能力。你的职责是：
1. 根据用户问题，精确检索知识库中的相关内容（含文本、表格、图表等）。
2. 基于检索到的上下文，给出准确、结构化、有据可查的回答。
3. 如果知识库中没有相关信息，明确告知用户，不要编造内容。
4. 引用来源时注明文档名称和页码，增强可信度。

检索策略：
- 简单问题：直接用 knowledge_base_search 检索。
- 复杂/多概念问题：先用 query_rewrite 生成子查询，再用 multi_round_search 多轮检索。
- 用户指定文件：改用 knowledge_base_search_with_filter。
- 单次检索结果不理想时，可调整查询后重试或切换多轮检索。

扩展技能策略：
- 数值计算：用 calculator，不要自己口算。
- 知识库无结果且需实时信息：用 web_search 补充。
- 检索到表格数据需统计分析：用 table_analyzer。
- 检索结果包含图像路径需理解图像：用 image_describer。
- 检索片段过长影响回答质量：先用 summarizer 压缩再组合。

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
        # 视觉模型（qwen-vl-max），用于 image_describer skill
        self._vl_llm = ChatOpenAI(
            model=settings.vl_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            request_timeout=settings.llm_request_timeout,
        )
        self._tools = (
            build_retrieval_tools(retriever, llm=self._llm)
            + build_skill_tools(llm=self._llm, vl_llm=self._vl_llm)
        )
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

        # 从消息历史中提取来源引用和工具调用步骤
        sources = self._extract_sources(output_messages)
        tool_steps = self._extract_tool_steps(output_messages)

        logger.info(
            f"[RAGAgent] 答案：{len(answer)} 字 | 来源：{len(sources)} | 工具调用：{len(tool_steps)}"
        )
        return {
            "answer": answer,
            "sources": sources,
            "tool_steps": tool_steps,
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
    def _extract_tool_steps(messages: list) -> list[dict]:
        """
        从 LangGraph 消息列表中提取工具调用步骤，用于前端展示思考过程。

        匹配逻辑：AIMessage.tool_calls 中的每个调用通过 tool_call_id
        与对应的 ToolMessage 关联，从而附上可读的结果摘要。
        """
        steps: list[dict] = []
        id_to_step: dict[str, dict] = {}

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    step = {
                        "tool": call["name"],
                        "args": call.get("args", {}),
                        "result_summary": "",
                    }
                    steps.append(step)
                    id_to_step[call["id"]] = step

            elif isinstance(msg, ToolMessage):
                step = id_to_step.get(msg.tool_call_id)
                if step is None:
                    continue
                content = msg.content if isinstance(msg.content, str) else ""
                step["result_summary"] = _summarize_tool_result(
                    step["tool"], content
                )

        return steps

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
