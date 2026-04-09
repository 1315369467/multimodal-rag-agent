"""
文档摘要 Skill
────────────────────────────────────────────────────────────────────────────
使用 LLM 对长文本片段进行结构化摘要压缩，支持自定义摘要长度和关注点。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from loguru import logger

# 超过此字符数才值得摘要，否则直接返回原文
_MIN_LEN_TO_SUMMARIZE = 300


def build_summarizer_tool(llm: Any):
    """
    返回文本摘要 LangChain Tool。

    参数
    ----
    llm : LangChain ChatModel 实例（需支持中文）。
    """

    @tool
    def summarizer(text: str, focus: str = "", max_length: int = 200) -> str:
        """
        对长文档片段进行摘要压缩，提取关键信息。

        适用场景：
        - 检索到的上下文过长，需要压缩后再组合回答
        - 对单篇文档做执行摘要（Executive Summary）
        - 提取特定主题下的关键要点

        参数
        ----
        text       : 需要摘要的原始文本（建议 300 字以上效果明显）。
        focus      : 可选，摘要关注点，例如 "财务数据" 或 "风险因素"。
                     留空时进行通用摘要。
        max_length : 期望摘要的最大字数，默认 200 字。
        """
        logger.info(
            f"[Skill:summarizer] text_len={len(text)}, "
            f"focus='{focus[:40]}', max_length={max_length}"
        )

        if not text.strip():
            return "输入文本为空，无法生成摘要。"

        if len(text) < _MIN_LEN_TO_SUMMARIZE:
            logger.info("[Skill:summarizer] 文本较短，直接返回原文")
            return text.strip()

        focus_line = f"\n关注点：{focus}" if focus.strip() else ""
        prompt = (
            f"请对以下文本生成一段简洁的中文摘要，要求：\n"
            f"1. 不超过 {max_length} 字\n"
            f"2. 保留所有关键数据、结论和实体名称\n"
            f"3. 使用客观、正式的语气{focus_line}\n\n"
            f"原文：\n{text}"
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            logger.info(f"[Skill:summarizer] 摘要 {len(summary)} 字")
            return summary
        except Exception as exc:
            logger.warning(f"[Skill:summarizer] 摘要失败：{exc}")
            return f"摘要生成失败：{exc}"

    return summarizer
