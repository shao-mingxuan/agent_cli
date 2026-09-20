"""示例插件工具。"""
from datetime import datetime

from langchain.tools import tool


@tool
def get_time() -> str:
    """获取当前时间。

    返回当前日期和时间，格式为 YYYY-MM-DD HH:MM:SS。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def translate(text: str, target_lang: str) -> str:
    """翻译文本到目标语言（示例占位）。

    Args:
        text: 要翻译的文本
        target_lang: 目标语言代码，如 en、ja、ko
    """
    return f"[翻译占位] {text} -> {target_lang}"
