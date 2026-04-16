"""
RAG Agent 的 LangChain 工具定义
──────────────────────────────────────────────────────────────────────────────
工具将检索子系统暴露给 LangChain Agent 循环。
Agent 在生成答案前可调用这些工具获取上下文。

工具清单
--------
- knowledge_base_search            : 文本知识库检索（稠密 + 稀疏 + 精排）
- knowledge_base_search_with_filter: 限定来源文件的文本检索
- image_search                     : 图像库跨模态检索（文本 query → 相关图片）
- query_rewrite                    : 将复杂问题改写为多个子查询（需要 LLM）
- multi_round_search               : 对多个子查询依次检索并合并去重结果（文本）
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from loguru import logger

from config.settings import get_settings

if TYPE_CHECKING:
    from src.retrieval.hybrid_retriever import HybridRetriever

_settings = get_settings()


# ---------------------------------------------------------------------------
# 工厂函数——在启动时绑定检索器，返回可用工具列表
# ---------------------------------------------------------------------------

def build_retrieval_tools(retriever: "HybridRetriever", llm: Any = None) -> list:
    """
    返回绑定到指定 HybridRetriever 的 LangChain 工具列表。

    参数
    ----
    retriever : 已初始化的 HybridRetriever。
    llm       : 可选，LangChain LLM 实例。提供后会额外注册
                query_rewrite 工具，用于复杂问题的查询改写。

    使用工厂模式（而非模块级全局变量）的原因：
    保持工具的可测试性，避免导入时的副作用。
    """

    @tool
    def knowledge_base_search(query: str) -> str:
        """
        在文本知识库中检索与查询相关的段落（不含图片）。

        当需要从已上传文档中获取事实性文本信息时使用此工具。
        返回包含来源元数据的相关段落 JSON 列表。
        如果用户需要查找图片/图表/示意图等视觉内容，请改用 image_search。

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
                "block_type": doc.metadata.get("block_type", "text"),
            }
            for i, (doc, _) in enumerate(docs[:5], start=1)
        ]
        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    @tool
    def multi_round_search(queries: str) -> str:
        """
        对多个查询依次检索知识库，合并并去重结果。

        通常与 query_rewrite 配合使用：
        先调用 query_rewrite 获得子查询列表，
        再将 rewritten_queries 序列化为 JSON 字符串传入此工具。

        参数
        ----
        queries : JSON 编码的查询字符串列表，
                  例如 '["子查询1", "子查询2", "子查询3"]'。
        """
        logger.info(f"[Tool:multi_round_search] queries='{queries[:120]}'")

        try:
            query_list = json.loads(queries)
            if not isinstance(query_list, list):
                query_list = [str(query_list)]
        except json.JSONDecodeError:
            # 容错：当作单个普通查询处理
            query_list = [queries]

        seen_keys: set[str] = set()
        merged: list[dict] = []
        rank = 1

        for q in query_list:
            if not q or not q.strip():
                continue
            docs = retriever.retrieve(q)
            for doc in docs:
                # 用内容前 120 字符作去重 key，避免重复片段
                dedup_key = doc.page_content[:120]
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                merged.append(
                    {
                        "rank": rank,
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page_num", "?"),
                        "block_type": doc.metadata.get("block_type", "text"),
                        "header_context": doc.metadata.get("header_context", ""),
                        "matched_query": q,
                    }
                )
                rank += 1

        if not merged:
            return json.dumps(
                {"results": [], "message": "多轮检索未找到相关内容。"},
                ensure_ascii=False,
            )

        return json.dumps(
            {"results": merged, "total_queries": len(query_list)},
            ensure_ascii=False,
            indent=2,
        )

    _image_top_k_default = _settings.image_top_k

    @tool
    def image_search(query: str, top_k: int = _image_top_k_default) -> str:
        """
        在图像向量库中用自然语言检索最相关的图片（跨模态检索）。

        当用户问题涉及图表、示意图、截图、曲线、架构图等视觉内容，
        或者需要展示某个概念对应的图片时使用此工具。
        底层使用 Qwen3-VL-Embedding 将文本与图片映射到同一向量空间。

        参数
        ----
        query  : 描述要找的图片的自然语言（例如「模型架构图」「训练损失曲线」）。
        top_k  : 返回图片数量（默认来自 settings.image_top_k，建议 3~10）。
        """
        logger.info(f"[Tool:image_search] query='{query[:80]}' top_k={top_k}")

        pairs = retriever.retrieve_images(query, k=top_k)
        if not pairs:
            return json.dumps(
                {"results": [], "message": "图片库为空或未找到相关图片。"},
                ensure_ascii=False,
            )

        results = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "source_path": meta.get("source_path", ""),
                    "file_name": meta.get("file_name", ""),
                    "alt_text": meta.get("alt_text", ""),
                    "caption": meta.get("caption", ""),
                    "source": meta.get("source", ""),
                    "block_type": "figure",
                }
            )

        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    tools = [
        knowledge_base_search,
        knowledge_base_search_with_filter,
        multi_round_search,
        image_search,
    ]

    # query_rewrite 依赖 LLM，仅在提供时注册
    if llm is not None:

        @tool
        def query_rewrite(query: str) -> str:
            """
            将复杂问题改写为多个不同角度的检索子查询，提升检索命中率。

            适用场景：
            - 问题涉及多个实体或概念
            - 问题表述模糊或过于口语化
            - 单次检索结果不理想时

            使用流程：
            1. 调用此工具生成 rewritten_queries 列表。
            2. 将列表序列化为 JSON 字符串，传给 multi_round_search。
            3. 基于合并后的检索结果生成答案。

            参数
            ----
            query : 原始用户问题。
            """
            logger.info(f"[Tool:query_rewrite] 原始问题='{query[:80]}'")

            prompt = (
                "请将以下问题改写为3个不同角度的检索查询，以提高知识库检索命中率。\n"
                f"原始问题：{query}\n\n"
                "要求：\n"
                "1. 每个查询从不同角度或使用不同关键词表达相同需求\n"
                "2. 保持语义一致，不引入无关内容\n"
                "3. 查询应简洁，适合关键词检索\n\n"
                '严格以 JSON 格式返回，例如：{"queries": ["查询1", "查询2", "查询3"]}\n'
                "只输出 JSON，不要任何其他内容。"
            )

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()

                # 兼容 LLM 输出 markdown 代码块的情况
                if "```" in content:
                    parts = content.split("```")
                    # 取第一个代码块内容
                    content = parts[1].lstrip("json").strip()

                data = json.loads(content)
                rewritten: list[str] = data.get("queries", [])
                if not rewritten:
                    raise ValueError("queries 列表为空")
            except Exception as exc:
                logger.warning(f"[Tool:query_rewrite] 改写失败，回退至原始问题：{exc}")
                rewritten = [query]

            logger.info(f"[Tool:query_rewrite] 生成 {len(rewritten)} 个子查询")
            return json.dumps(
                {"original_query": query, "rewritten_queries": rewritten},
                ensure_ascii=False,
            )

        tools.append(query_rewrite)

    return tools
