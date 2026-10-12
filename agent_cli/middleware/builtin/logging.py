"""L6 中间件 - 对话日志。"""

import logging
from typing import Any

from ...utils.logger import get_logger
from ..types import GuardResult


class LoggingMiddleware:
    """以 DEBUG 级别记录每轮对话的输入/输出长度。

    DEBUG 级别默认不输出，避免刷屏；将 logger 级别调低即可看到。
    """

    def __init__(self, logger: logging.Logger | None = None, max_preview: int = 200):
        self._logger = logger or get_logger("agent_cli.middleware.logging")
        self._max_preview = max_preview

    def _preview(self, text: str) -> str:
        text = text.replace("\n", " ")
        return text[: self._max_preview] + ("..." if len(text) > self._max_preview else "")

    def before_run(self, user_input: str) -> GuardResult:
        self._logger.debug("run start | input(%d chars): %s", len(user_input), self._preview(user_input))
        return GuardResult()

    def after_run(self, user_input: str, response: str) -> GuardResult:
        self._logger.debug("run done | output(%d chars): %s", len(response), self._preview(response))
        return GuardResult()
