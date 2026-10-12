"""L2 提示词 - 切换/版本/回滚。"""

from typing import Any

from .builtin import BUILTIN_PROMPTS


class PromptManager:
    """管理一组命名提示词，支持切换与回滚。

    current 指向当前生效的提示词名；set_current 会记录历史，
    rollback 可回到上一个提示词。
    """

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        current: str | None = None,
    ):
        self._prompts: dict[str, str] = dict(prompts or {})
        self._history: list[str] = []
        if current is not None:
            if current not in self._prompts:
                raise KeyError(f"未知提示词: {current}")
            self._current = current
        else:
            self._current = next(iter(self._prompts), None)

    def get(self, name: str | None = None) -> str:
        """返回指定（或当前）提示词文本。"""
        name = name or self._current
        if name is None:
            return ""
        try:
            return self._prompts[name]
        except KeyError as exc:
            raise KeyError(f"未知提示词: {name}") from exc

    def register(self, name: str, prompt: str) -> None:
        self._prompts[name] = prompt

    def register_many(self, prompts: dict[str, str]) -> None:
        self._prompts.update(prompts)

    def list(self) -> list[str]:
        return list(self._prompts)

    @property
    def current(self) -> str | None:
        return self._current

    def set_current(self, name: str) -> str:
        """切换到指定提示词并记录历史，返回提示词文本。"""
        if name not in self._prompts:
            raise KeyError(f"未知提示词: {name}")
        if self._current is not None:
            self._history.append(self._current)
        self._current = name
        return self._prompts[name]

    def rollback(self) -> str | None:
        """回滚到上一个提示词；无历史时返回 None。"""
        if not self._history:
            return None
        self._current = self._history.pop()
        return self._current


def create_prompt_manager() -> PromptManager:
    """创建预置内置 prompts 的管理器。"""
    return PromptManager(prompts=dict(BUILTIN_PROMPTS), current="default")
