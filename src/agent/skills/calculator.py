"""
计算器 Skill
────────────────────────────────────────────────────────────────────────────
使用 Python AST 白名单安全执行数学表达式，不调用 eval/exec，
防止任意代码注入。
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool
from loguru import logger

# 允许的二元 / 一元运算符
_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# 允许调用的数学函数
_MATH_FUNCS: dict[str, Any] = {
    name: getattr(math, name)
    for name in (
        "sqrt", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "ceil", "floor", "fabs", "factorial",
    )
}
_MATH_FUNCS["abs"] = abs
_MATH_FUNCS["round"] = round


def _safe_eval(node: ast.expr) -> float:
    """递归地对 AST 节点求值，仅允许白名单内的操作。"""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量类型：{type(node.value)}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(op_fn(left, right))

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return float(op_fn(_safe_eval(node.operand)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("不支持复杂函数调用")
        fn = _MATH_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"不允许的函数：{node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return float(fn(*args))

    if isinstance(node, ast.Name):
        constants = {"pi": math.pi, "e": math.e, "inf": math.inf}
        if node.id in constants:
            return constants[node.id]
        raise ValueError(f"未知变量：{node.id}")

    raise ValueError(f"不支持的 AST 节点：{type(node).__name__}")


def build_calculator_tool():
    """返回计算器 LangChain Tool。"""

    @tool
    def calculator(expression: str) -> str:
        """
        安全地计算数学表达式，返回数值结果。

        适用场景：
        - 文档中财务数字的加减乘除、百分比换算
        - 表格数据的统计汇总（可先用 table_analyzer 提取数值）
        - 任何需要精确计算的数值问题

        支持：+、-、*、/、//、%、**（乘方）以及
        sqrt、log、sin/cos/tan、ceil/floor、abs、round 等数学函数，
        常量 pi、e、inf。

        参数
        ----
        expression : 数学表达式字符串，例如 '(12 + 8) * 3.14'
                     或 'sqrt(144) + log(100, 10)'
        """
        logger.info(f"[Skill:calculator] expr='{expression}'")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree.body)
            # 整数结果去掉小数点
            display = int(result) if result == int(result) else result
            logger.info(f"[Skill:calculator] result={display}")
            return f"计算结果：{expression} = {display}"
        except ZeroDivisionError:
            return "错误：除数不能为零。"
        except Exception as exc:
            logger.warning(f"[Skill:calculator] 计算失败：{exc}")
            return f"计算失败：{exc}"

    return calculator
