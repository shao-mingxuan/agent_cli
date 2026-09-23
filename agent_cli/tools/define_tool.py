"""L3 工具 - define_tool() 辅助函数。"""

from langchain.tools import tool as _tool


def define_tool(name: str, description: str):
    """定义一个工具的装饰器，封装 LangChain 的 @tool。

    Args:
        name: 工具名称
        description: 工具描述
    """

    def decorator(func):
        func.__name__ = name
        func.__doc__ = description
        return _tool(func)

    return decorator
