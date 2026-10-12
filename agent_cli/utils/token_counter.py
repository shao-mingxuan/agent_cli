"""L6 工具 - token 计数。"""

import threading
from typing import Callable

try:
    import tiktoken
except ImportError:  # pragma: no cover - tiktoken 为可选依赖
    tiktoken = None

_ENCODER_CACHE: dict[str, object | None] = {}
_LOAD_TIMEOUT = 3.0


def _load_encoder(model: str = ""):
    """在子线程中加载编码器，超时则放弃（避免 tiktoken 联网下载卡死）。"""
    result: dict[str, object | None] = {"enc": None}

    def _run():
        try:
            enc = tiktoken.encoding_for_model(model or "gpt-4o")
            result["enc"] = enc
        except Exception:
            try:
                result["enc"] = tiktoken.get_encoding("cl100k_base")
            except Exception:
                result["enc"] = None

    if tiktoken is None:
        return None
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_LOAD_TIMEOUT)
    if thread.is_alive():
        return None
    return result["enc"]


def _get_encoder(model: str = ""):
    if model not in _ENCODER_CACHE:
        _ENCODER_CACHE[model] = _load_encoder(model)
    return _ENCODER_CACHE[model]


def count_tokens(text: str, model: str = "") -> int:
    """估算文本 token 数。

    tiktoken 可用（且已缓存）时精确计数；否则回退为字符数/4 估算。
    所有可能联网下载的路径都在子线程中带超时执行，不会阻塞调用方。
    """
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(messages: list, model: str = "") -> int:
    """估算消息列表总 token 数（含 tool_calls 内容）。"""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += count_tokens(content, model)
        for tc in getattr(msg, "tool_calls", None) or []:
            if isinstance(tc, dict):
                label = tc.get("name", "") + str(tc.get("args", {}))
            else:
                label = str(getattr(tc, "name", "")) + str(getattr(tc, "args", {}))
            total += count_tokens(label, model)
    return total


def make_token_counter(model: str = "") -> Callable[[str], int]:
    """返回可复用的 token 计数回调，供 WorkingMemory 等场景使用。"""
    return lambda text: count_tokens(text, model)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float = 0.000001,
    output_price: float = 0.000002,
) -> float:
    """按每 token 单价估算费用（默认美元）。

    默认单价是占位值，真实单价应由具体模型的定价表覆盖。
    """
    return input_tokens * input_price + output_tokens * output_price
