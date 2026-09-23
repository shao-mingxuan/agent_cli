"""L1 编排核心 - AgentState, AgentEvent 类型定义。"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage


class StepType:
    """Agent 循环步骤类型。"""

    THINK = "think"
    ACT = "act"
    RESPOND = "respond"
    THINKING = "thinking"
    TOKEN = "token"
    GUARD = "guard"
    APPROVE = "approve"


@dataclass
class AgentState:
    """Agent 运行时状态。"""

    messages: list[BaseMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Agent 输出事件。"""

    step: str
    content: str = ""
    type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
