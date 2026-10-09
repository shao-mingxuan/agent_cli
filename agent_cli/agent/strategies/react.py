"""L1 编排核心 - 手动 ReAct 循环策略（替代 LangGraph create_agent）。"""

from typing import Any, Iterator

from langchain_core.messages import SystemMessage, ToolMessage

from ..approval import MAX_TOOL_FAILURES, is_sensitive_tool, is_tool_error
from ..types import AgentEvent, StepType


def _to_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def manual_react_loop(
    orch, config: dict, pending_tool_calls: dict
) -> Iterator[dict]:
    """手写 ReAct 循环：model(bind_tools).invoke → tool_calls → 执行工具 → 回填。

    事件序列与 run_stream_loop 兼容：
    THINKING → THINK → (APPROVE) → ACT → ... → TOKEN → 返回 state dict。

    返回 state 结构（供 finalize_response 消费）：
    - respond_chunks / aborted / consecutive_failures / memory_len_before
    """
    respond_chunks: list[str] = []
    consecutive_failures = 0
    aborted = False
    _memory_len_before = len(orch.memory.get_messages())
    max_steps = getattr(orch, "_react_max_steps", 8)

    model = orch.model.bind_tools(orch.tools)

    for _step in range(max_steps):
        yield AgentEvent(step=StepType.THINKING, content="")

        messages = [SystemMessage(content=orch.system_prompt)] + orch.memory.get_messages()

        try:
            ai_msg = model.invoke(messages)
        except Exception as e:
            error_text = f"[模型调用失败] {type(e).__name__}: {e}"
            respond_chunks.append(error_text)
            yield AgentEvent(step=StepType.RESPOND, content=error_text)
            aborted = True
            break

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            content = _to_text(ai_msg.content)
            if content:
                orch.memory.add_ai(content)
                yield AgentEvent(step=StepType.TOKEN, content=content)
                respond_chunks.append(content)
            break

        orch.memory.add_message(ai_msg)

        pending: list[dict] = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            info = orch._tool_lookup.get(name)
            category = info.category if info else ""
            pending.append(
                {
                    "id": tc["id"],
                    "name": name,
                    "args": args,
                    "source": info.source if info else "unknown",
                    "category": category,
                    "sensitive": is_sensitive_tool(
                        {"name": name, "category": category}
                    ),
                }
            )

        yield AgentEvent(
            step=StepType.THINK,
            content="",
            metadata={
                "tool_details": [
                    {
                        "name": p["name"],
                        "source": p["source"],
                        "category": p["category"],
                        "args": p["args"],
                    }
                    for p in pending
                ]
            },
        )

        rejected_ids: set[str] = set()
        if any(p["sensitive"] for p in pending):
            yield AgentEvent(
                step=StepType.APPROVE,
                content="",
                metadata={"tool_calls": pending},
            )
            approved_list = orch._approval_callback(pending)
            rejected_ids = {
                tc["id"] for tc, ok in zip(pending, approved_list) if not ok
            }

        for tc_call in pending:
            tc_id = tc_call["id"]
            name = tc_call["name"]
            if tc_id in rejected_ids:
                content = "用户拒绝了此工具调用。"
                orch.memory.add_message(
                    ToolMessage(content=content, name=name, tool_call_id=tc_id)
                )
                yield AgentEvent(
                    step=StepType.ACT,
                    content=content,
                    metadata={
                        "tool_name": name,
                        "tool_source": tc_call["source"],
                        "tool_category": tc_call["category"],
                        "tool_args": tc_call["args"],
                        "tool_call_id": tc_id,
                        "rejected": True,
                    },
                )
                consecutive_failures += 1
                continue

            info = orch._tool_lookup.get(name)
            try:
                raw = info.tool.invoke(tc_call["args"]) if info else f"未知工具: {name}"
                tool_content = _to_text(raw)
            except Exception as e:
                tool_content = f"工具执行失败: {e}"

            orch.memory.add_message(
                ToolMessage(content=tool_content, name=name, tool_call_id=tc_id)
            )
            yield AgentEvent(
                step=StepType.ACT,
                content=tool_content,
                metadata={
                    "tool_name": name,
                    "tool_source": tc_call["source"],
                    "tool_category": tc_call["category"],
                    "tool_args": tc_call["args"],
                    "tool_call_id": tc_id,
                },
            )

            if is_tool_error(tool_content):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        if consecutive_failures >= MAX_TOOL_FAILURES:
            aborted = True
            break

    return {
        "respond_chunks": respond_chunks,
        "aborted": aborted,
        "consecutive_failures": consecutive_failures,
        "memory_len_before": _memory_len_before,
    }
