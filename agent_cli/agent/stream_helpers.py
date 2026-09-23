"""L1 编排 - 追问兜底 Helper 函数。"""

from langchain_core.messages import AIMessage


def last_model_output_is_question(orch, config: dict) -> bool:
    """检查对话中最后一条 AIMessage 的纯文本是否以问号结尾。

    用于追问兜底：如果模型正在向用户澄清（以问号结尾的追问），
    同一回合不应再继续执行任何工具调用。
    """
    state = orch.agent.get_state(config)
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip().endswith(("?", "？"))
            break
    return False


def append_question_text_to_response(
    orch, config: dict, respond_chunks: list[str]
) -> None:
    """把模型追问文本追加到 respond_chunks。

    追问兜底触发时，模型可能在同一个 AIMessage 里既输出追问文本又发起了
    工具调用，此时文本已被跳过未累积。这里从 checkpoint 中找回它，
    确保 break 后追问内容能作为最终回复输出给用户。
    """
    state = orch.agent.get_state(config)
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                respond_chunks.append(content)
            break
