"""L3 技能 - Skill 定义与装饰器。"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Skill:
    """复合技能：预配置的 prompt + 工具子集 + 可选处理逻辑。"""

    name: str
    description: str
    system_prompt: str
    tool_allowlist: list[str] | None = None
    preprocess: Callable[[str], str] | None = None
    postprocess: Callable[[str], str] | None = None


def define_skill(
    name: str,
    description: str,
    system_prompt: str,
    tool_allowlist: list[str] | None = None,
):
    """装饰器：将函数标记为 Skill，用于带自定义处理逻辑的技能。

    被装饰函数体本身不会被执行，仅用于承载 preprocess / postprocess 逻辑。
    """

    def decorator(func: Callable) -> Skill:
        return Skill(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tool_allowlist=tool_allowlist,
            preprocess=getattr(func, "preprocess", None),
            postprocess=getattr(func, "postprocess", None),
        )

    return decorator
