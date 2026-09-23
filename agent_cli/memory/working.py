"""L4 记忆 - 当前会话（内存）。"""
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

DEFAULT_MAX_MESSAGES = 50
DEFAULT_MAX_TOKENS = 4000
DEFAULT_COMPRESSION_THRESHOLD = 20
DEFAULT_KEEP_RECENT = 10


class WorkingMemory:
    """维护当前会话的消息列表，实现多轮对话上下文。

    支持两层上下文管理：
    - L1 滑动窗口：消息数超过 max_messages 时，淘汰最早的非 system 消息。
    - L2 摘要压缩：token 数超过 max_tokens 时，用 LLM 将旧消息压缩为摘要。
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_tokens: int | None = None,
        compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        summarizer: Callable[[str], str] | None = None,
    ):
        self._messages: list[BaseMessage] = []
        self._max_messages = max_messages
        self._max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self._compression_threshold = compression_threshold
        self._keep_recent = keep_recent
        self._summarizer = summarizer
        self._encoder = None

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @max_messages.setter
    def max_messages(self, value: int) -> None:
        self._max_messages = value
        self._maybe_trim()

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value
        self._maybe_compress()
        self._maybe_trim()

    @property
    def compression_enabled(self) -> bool:
        return self._summarizer is not None

    def add_human(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._maybe_compress()
        self._maybe_trim()

    def add_ai(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        self._maybe_compress()
        self._maybe_trim()

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._maybe_compress()
        self._maybe_trim()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._messages.extend(messages)
        self._maybe_compress()
        self._maybe_trim()

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def rollback_to(self, n: int) -> None:
        """回滚消息列表到长度 n，丢弃后面的消息。"""
        if len(self._messages) > n:
            del self._messages[n:]

    def clear(self) -> None:
        self._messages.clear()

    def _get_encoder(self):
        """Lazy init tiktoken encoder，失败时降级为字符数估算。"""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                self._encoder = False
        return self._encoder

    def _estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """估算消息列表的 token 数。tiktoken 不可用时降级为字符数/4。"""
        enc = self._get_encoder()
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if enc:
                total += len(enc.encode(content))
            else:
                total += len(content) // 4
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tc_text = tc.get("name", "") + str(tc.get("args", {}))
                    if enc:
                        total += len(enc.encode(tc_text))
                    else:
                        total += len(tc_text) // 4
        return total

    def _serialize_messages(self, messages: list[BaseMessage]) -> str:
        """将消息列表序列化为可读文本，用于摘要输入。"""
        lines = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                tc_str = "; ".join(
                    f"调用 {tc['name']}({tc.get('args', {})})" for tc in tool_calls
                )
                content = f"{content} [{tc_str}]" if content else tc_str
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _maybe_compress(self) -> None:
        """token 预算超限时触发摘要压缩。"""
        if self._summarizer is None:
            return

        non_system = [m for m in self._messages if not isinstance(m, SystemMessage)]
        if len(non_system) < self._compression_threshold:
            return

        if self._estimate_tokens(self._messages) <= self._max_tokens:
            return

        # 分离：保留的 system prompt、旧摘要（可压缩）、普通消息
        system_msgs = [
            m for m in self._messages
            if isinstance(m, SystemMessage) and "[历史摘要]" not in m.content
        ]
        old_summaries = [
            m for m in self._messages
            if isinstance(m, SystemMessage) and "[历史摘要]" in m.content
        ]
        other_msgs = [m for m in self._messages if not isinstance(m, SystemMessage)]

        keep = min(self._keep_recent, len(other_msgs))
        to_compress = old_summaries + (other_msgs[:-keep] if keep > 0 else other_msgs)
        recent_msgs = other_msgs[-keep:] if keep > 0 else []

        if len(to_compress) < 2:
            return

        text = self._serialize_messages(to_compress)
        try:
            summary_text = self._summarizer(text)
        except Exception:
            return

        self._messages = (
            system_msgs
            + [SystemMessage(content=f"[历史摘要]\n{summary_text}")]
            + recent_msgs
        )

    def _maybe_trim(self) -> None:
        """滑动窗口裁剪：保留所有 SystemMessage（含摘要）+ 最近的消息。"""
        if len(self._messages) <= self._max_messages:
            return

        system_msgs = [m for m in self._messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in self._messages if not isinstance(m, SystemMessage)]

        max_other = self._max_messages - len(system_msgs)
        max_other = max(max_other, 0)

        if len(other_msgs) > max_other:
            other_msgs = other_msgs[-max_other:] if max_other > 0 else []

        self._messages = system_msgs + other_msgs
