"""L0 CLI - 共享工具。"""

from rich.console import Console

console = Console()


def make_tag(source: str, category: str, name: str) -> str:
    """构建 source/category/name 标签字符串。"""
    if category:
        return f"{source}/{category}/{name}"
    return f"{source}/{name}"
