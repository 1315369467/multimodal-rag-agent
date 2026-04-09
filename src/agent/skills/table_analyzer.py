"""
表格分析 Skill
────────────────────────────────────────────────────────────────────────────
解析 Markdown / CSV 格式表格，用 numpy 计算统计指标（均值、求和、最大最小等）。
不依赖 pandas，仅使用项目已有的 numpy。
"""
from __future__ import annotations

import io
import json
import re
from typing import Any

import numpy as np
from langchain_core.tools import tool
from loguru import logger


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def _parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]]:
    """
    解析 Markdown 表格，返回 (headers, rows)。
    支持含对齐行（ :--- | :---: | ---: ）的标准 Markdown 表格。
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]

    if not table_lines:
        raise ValueError("未检测到 Markdown 表格（行首需有 '|'）")

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    headers = split_row(table_lines[0])
    # 跳过对齐行（全为 -、: 字符）
    data_lines = [
        l for l in table_lines[1:]
        if not re.fullmatch(r"[\|\s\-:]+", l)
    ]
    rows = [split_row(l) for l in data_lines]
    return headers, rows


def _parse_csv_table(text: str) -> tuple[list[str], list[list[str]]]:
    """解析 CSV 格式表格。"""
    reader = list(
        line.split(",") for line in text.strip().splitlines() if line.strip()
    )
    if not reader:
        raise ValueError("CSV 内容为空")
    headers = [c.strip() for c in reader[0]]
    rows = [[c.strip() for c in r] for r in reader[1:]]
    return headers, rows


def _try_parse(text: str) -> tuple[list[str], list[list[str]]]:
    """自动检测格式（Markdown 优先，否则尝试 CSV）。"""
    if "|" in text:
        return _parse_markdown_table(text)
    return _parse_csv_table(text)


def _col_stats(values: list[str]) -> dict[str, Any] | None:
    """
    对一列字符串值尝试转换为浮点数并计算统计量。
    若列不是数值型返回 None。
    """
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v.replace(",", "").replace("，", "")))
        except (ValueError, AttributeError):
            pass
    if not nums:
        return None
    arr = np.array(nums)
    return {
        "count": int(arr.size),
        "sum": round(float(arr.sum()), 4),
        "mean": round(float(arr.mean()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "std": round(float(arr.std()), 4),
    }


# ---------------------------------------------------------------------------
# Tool 工厂
# ---------------------------------------------------------------------------

def build_table_analyzer_tool():
    """返回表格分析 LangChain Tool。"""

    @tool
    def table_analyzer(table_text: str, operation: str = "stats") -> str:
        """
        解析并分析 Markdown 或 CSV 格式的表格数据，输出统计结果。

        适用场景：
        - 对检索结果中的财务报表、数据汇总表做统计分析
        - 计算各列的总和、均值、最大/最小值、标准差
        - 对比表格中不同行或列的数值

        参数
        ----
        table_text : Markdown 表格或 CSV 格式的表格文本。
                     Markdown 示例：
                       | 季度 | 收入 | 成本 |
                       |------|------|------|
                       | Q1   | 100  | 60   |
        operation  : 分析操作，目前支持 'stats'（默认，所有数值列的统计量）。
        """
        logger.info(f"[Skill:table_analyzer] op={operation}, length={len(table_text)}")

        try:
            headers, rows = _try_parse(table_text)
        except Exception as exc:
            return f"表格解析失败：{exc}\n请确保输入为标准 Markdown 表格或 CSV 格式。"

        if not rows:
            return "表格无数据行。"

        n_cols = len(headers)
        # 转置为列
        col_data: dict[str, list[str]] = {
            h: [r[i] if i < len(r) else "" for r in rows]
            for i, h in enumerate(headers)
        }

        output_lines = [
            f"表格概况：{len(rows)} 行 × {n_cols} 列",
            f"列名：{', '.join(headers)}",
            "",
            "数值列统计：",
        ]

        has_numeric = False
        for col_name, values in col_data.items():
            stats = _col_stats(values)
            if stats is None:
                continue
            has_numeric = True
            output_lines.append(
                f"  【{col_name}】"
                f"  求和={stats['sum']}"
                f"  均值={stats['mean']}"
                f"  最小={stats['min']}"
                f"  最大={stats['max']}"
                f"  标准差={stats['std']}"
                f"  计数={stats['count']}"
            )

        if not has_numeric:
            output_lines.append("  （未发现数值列）")

        return "\n".join(output_lines)

    return table_analyzer
