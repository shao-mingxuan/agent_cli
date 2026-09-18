"""L6 中间件 - AgentMiddleware 接口。"""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GuardResult:
    """中间件检测结果。"""

    blocked: bool = False
    risk_level: str = "safe"
    message: str = ""
    findings: list[dict] = field(default_factory=list)


class AgentMiddleware(Protocol):
    """中间件协议：在 agent 执行前后插入逻辑。"""

    def before_run(self, user_input: str) -> GuardResult:
        """前置检测：返回 GuardResult，blocked=True 则拦截不调模型。"""
        ...

    def after_run(self, user_input: str, response: str) -> GuardResult:
        """后置检测：对模型回复做安全检测。"""
        ...
