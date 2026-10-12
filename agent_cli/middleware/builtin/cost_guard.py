"""L6 中间件 - 费用统计与费用上限控制。"""

from threading import Lock

from ...utils.token_counter import count_tokens
from ..types import GuardResult


class CostGuardMiddleware:
    """累计每次对话的 token 用量与估算费用。

    - before_run 计入用户输入 token
    - after_run 计入模型回复 token
    - max_cost 设置后，费用超限会拦截后续回复
    - stats() 返回累计统计，reset() 清零（如新会话开始时）
    """

    def __init__(
        self,
        input_price: float = 0.000001,
        output_price: float = 0.000002,
        max_cost: float | None = None,
        model: str = "",
    ):
        self._input_price = input_price
        self._output_price = output_price
        self._max_cost = max_cost
        self._model = model
        self._lock = Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0

    def before_run(self, user_input: str) -> GuardResult:
        with self._lock:
            self._calls += 1
            self._input_tokens += count_tokens(user_input, self._model)
        return GuardResult()

    def after_run(self, user_input: str, response: str) -> GuardResult:
        with self._lock:
            self._output_tokens += count_tokens(response, self._model)
            cost = self.total_cost()
        if self._max_cost is not None and cost > self._max_cost:
            return GuardResult(
                blocked=True,
                risk_level="warning",
                message=f"累计估算费用 {cost:.4f} 已超过上限 {self._max_cost:.4f}，已拦截本次回复。",
            )
        return GuardResult()

    def total_cost(self) -> float:
        return (
            self._input_tokens * self._input_price
            + self._output_tokens * self._output_price
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self._calls,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "estimated_cost": round(self.total_cost(), 6),
            }

    def reset(self) -> None:
        with self._lock:
            self._input_tokens = 0
            self._output_tokens = 0
            self._calls = 0
