"""
Agent Skills 包
────────────────────────────────────────────────────────────────────────────
对外暴露 build_skill_tools 工厂函数，统一注册所有扩展技能工具。

技能清单
--------
- calculator      : 安全数学表达式求值
- web_search      : DuckDuckGo 实时网络搜索
- table_analyzer  : Markdown/CSV 表格统计分析
- image_describer : Qwen-VL 图像内容描述（需 vl_llm）
- summarizer      : LLM 文本摘要压缩（需 llm）
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from .calculator import build_calculator_tool
from .image_describer import build_image_describer_tool
from .summarizer import build_summarizer_tool
from .table_analyzer import build_table_analyzer_tool
from .web_search import build_web_search_tool


def build_skill_tools(llm: Any = None, vl_llm: Any = None) -> list:
    """
    构建并返回所有可用的 Skill 工具列表。

    参数
    ----
    llm    : 可选，文本 LLM（ChatOpenAI 等）。提供后注册 summarizer。
    vl_llm : 可选，多模态视觉 LLM（指向 qwen-vl-max）。提供后注册 image_describer。

    返回
    ----
    LangChain Tool 列表，可直接传入 create_react_agent 的 tools 参数。
    """
    tools = []

    # 无需外部依赖的基础 skills
    tools.append(build_calculator_tool())
    logger.debug("[Skills] calculator 已注册")

    tools.append(build_web_search_tool())
    logger.debug("[Skills] web_search 已注册")

    tools.append(build_table_analyzer_tool())
    logger.debug("[Skills] table_analyzer 已注册")

    # 依赖 LLM 的 skills
    if llm is not None:
        tools.append(build_summarizer_tool(llm))
        logger.debug("[Skills] summarizer 已注册")
    else:
        logger.debug("[Skills] summarizer 跳过（未提供 llm）")

    if vl_llm is not None:
        tools.append(build_image_describer_tool(vl_llm))
        logger.debug("[Skills] image_describer 已注册")
    else:
        logger.debug("[Skills] image_describer 跳过（未提供 vl_llm）")

    logger.info(f"[Skills] 共注册 {len(tools)} 个 skill 工具")
    return tools
