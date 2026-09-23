"""L3 工具 - 数学计算工具（基于 AST 白名单的安全求值）。"""

import ast
import math
import operator
from typing import Any

from langchain.tools import tool

_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_CALLS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
    "len": len,
}

_ALLOWED_MATH_ATTRS: set[str] = {
    "pi",
    "e",
    "tau",
    "inf",
    "nan",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "log10",
    "log2",
    "exp",
    "ceil",
    "floor",
    "fabs",
    "factorial",
    "gcd",
    "isqrt",
    "pow",
    "radians",
    "degrees",
    "hypot",
}

_ALLOWED_NAMES: set[str] = {"True", "False", "None"}


class _SecurityError(Exception):
    """表达式包含不允许的节点类型。"""


def _eval_node(node: ast.AST) -> Any:
    """递归求值 AST 节点，仅允许白名单节点。"""
    # ── 数值 ──
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, str, bytes, bool, type(None))):
            if isinstance(node.value, str) and not node.value.isdigit():
                # 拒绝非数字字符串字面量，防止信息泄露
                raise _SecurityError(f"不支持的字符串常量: {node.value!r}")
            return node.value
        raise _SecurityError(f"不支持的常量类型: {type(node.value).__name__}")

    # ── 名称 ──
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return {"True": True, "False": False, "None": None}[node.id]
        if node.id in _ALLOWED_CALLS:
            return _ALLOWED_CALLS[node.id]
        raise _SecurityError(f"不允许的名称: {node.id}")

    # ── math 属性 ──
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr in _ALLOWED_MATH_ATTRS:
                return getattr(math, node.attr)
            raise _SecurityError(f"不允许的 math 属性: {node.attr}")
        raise _SecurityError("不允许的属性访问")

    # ── 二元运算 ──
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise _SecurityError(f"不允许的二元运算符: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[op_type](left, right)

    # ── 一元运算 ──
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise _SecurityError(f"不允许的一元运算符: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _ALLOWED_UNARY_OPS[op_type](operand)

    # ── 函数调用 ──
    if isinstance(node, ast.Call):
        # 仅支持白名单函数和 math 方法
        func = _eval_node(node.func)
        if callable(func):
            args = [_eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: _eval_node(kw.value) for kw in node.keywords}
            if kwargs:
                raise _SecurityError("不支持关键字参数")
            return func(*args)
        raise _SecurityError("不允许的函数调用")

    # ── 字面量容器 ──
    if isinstance(node, ast.List):
        return [_eval_node(elt) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    raise _SecurityError(f"不允许的 AST 节点: {type(node).__name__}")


def _safe_eval(expression: str) -> Any:
    """解析并安全求值数学表达式。"""
    if not isinstance(expression, str):
        raise _SecurityError("表达式必须是字符串")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise _SecurityError(f"语法错误: {e}") from e

    return _eval_node(tree)


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除、括号、math 函数等。

    Args:
        expression: 数学表达式，如 "3 * 7 + 2" 或 "math.sqrt(16)"
    """
    try:
        result = _safe_eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"
