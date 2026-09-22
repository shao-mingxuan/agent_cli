"""L4 记忆 - 当前会话（内存）。"""
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class WorkingMemory:
    """维护当前会话的消息列表，实现多轮对话上下文。"""

    def __init__(self):
        self._messages: list[BaseMessage] = []

    def add_human(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_ai(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._messages.extend(messages)

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def rollback_to(self, n: int) -> None:
        """回滚消息列表到长度 n，丢弃后面的消息。"""
        if len(self._messages) > n:
            del self._messages[n:]

    def clear(self) -> None:
        self._messages.clear()
