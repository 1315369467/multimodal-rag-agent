"""
网络搜索 Skill
────────────────────────────────────────────────────────────────────────────
直接使用 duckduckgo-search 6+ 的 DDGS 接口搜索互联网，
绕过 langchain_community 对旧版 API 的依赖。
"""
from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger


def build_web_search_tool():
    """返回网络搜索 LangChain Tool。"""

    try:
        from ddgs import DDGS
        _available = True
    except ImportError:
        DDGS = None
        _available = False
        logger.warning(
            "[Skill:web_search] duckduckgo-search 未安装，"
            "请执行 `pip install duckduckgo-search` 后重启。"
        )

    @tool
    def web_search(query: str) -> str:
        """
        在互联网上搜索最新信息，返回摘要结果。

        适用场景：
        - 知识库中没有相关文档，但需要实时或近期数据
        - 查询时事、新闻、产品发布等动态信息
        - 补充知识库检索结果中缺失的背景知识

        注意：仅在知识库检索无结果或问题明确需要实时信息时调用。

        参数
        ----
        query : 搜索关键词或自然语言问题，建议简洁（20 字以内效果最佳）。
        """
        logger.info(f"[Skill:web_search] query='{query[:80]}'")

        if not _available:
            return (
                "网络搜索功能不可用：请先安装 `duckduckgo-search` 包，"
                "命令：pip install duckduckgo-search"
            )

        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=5))

            if not hits:
                return "未找到相关结果。"

            lines = []
            for i, h in enumerate(hits, 1):
                title = h.get("title", "")
                body = h.get("body", "")
                href = h.get("href", "")
                lines.append(f"{i}. **{title}**\n   {body}\n   来源：{href}")

            result = "\n\n".join(lines)
            logger.info(f"[Skill:web_search] 返回 {len(hits)} 条结果")
            return result

        except Exception as exc:
            logger.warning(f"[Skill:web_search] 搜索失败：{exc}")
            return f"搜索失败：{exc}"

    return web_search
