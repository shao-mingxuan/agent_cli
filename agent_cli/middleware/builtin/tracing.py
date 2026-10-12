"""L6 中间件 - 链路追踪（每轮耗时）。"""

import logging
import time

from ...utils.logger import get_logger
from ..types import GuardResult


class TracingMiddleware:
    """记录每一轮"输入→输出"的耗时。

    - last_duration_ms 记录最近一轮耗时
    - duration_list 保存最近 N 轮耗时，可用于 /config 展示
    """

    def __init__(self, logger: logging.Logger | None = None, keep: int = 100):
        self._logger = logger or get_logger("agent_cli.middleware.tracing")
        self._keep = keep
        self._start: float | None = None
        self._last_duration_ms: float = 0.0
        self._durations: list[float] = []

    def before_run(self, user_input: str) -> GuardResult:
        self._start = time.perf_counter()
        return GuardResult()

    def after_run(self, user_input: str, response: str) -> GuardResult:
        if self._start is None:
            return GuardResult()
        elapsed = (time.perf_counter() - self._start) * 1000
        self._start = None
        self._last_duration_ms = elapsed
        self._durations.append(elapsed)
        if len(self._durations) > self._keep:
            self._durations = self._durations[-self._keep :]
        self._logger.debug("run took %.1f ms", elapsed)
        return GuardResult()

    @property
    def last_duration_ms(self) -> float:
        return self._last_duration_ms

    @property
    def duration_list(self) -> list[float]:
        return list(self._durations)

    def reset(self) -> None:
        self._start = None
        self._last_duration_ms = 0.0
        self._durations = []
