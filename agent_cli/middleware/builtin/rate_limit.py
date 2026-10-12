"""L6 中间件 - 速率限制。"""

import time
from collections import deque
from threading import Lock

from ..types import GuardResult


class RateLimitMiddleware:
    """按固定时间窗口限制请求频率（滑动窗口，内存态）。

    默认限制 60 次/60 秒，超过返回 blocked 结果让上层提示用户稍后重试。
    """

    def __init__(self, limit: int = 60, period: float = 60.0):
        self._limit = limit
        self._period = period
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def before_run(self, user_input: str) -> GuardResult:
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > self._period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._limit:
                wait = self._period - (now - self._timestamps[0])
                return GuardResult(
                    blocked=True,
                    risk_level="warning",
                    message=f"请求过于频繁，请 {max(wait, 1):.0f}s 后重试。",
                )
            self._timestamps.append(now)
        return GuardResult()

    def after_run(self, user_input: str, response: str) -> GuardResult:
        return GuardResult()

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
