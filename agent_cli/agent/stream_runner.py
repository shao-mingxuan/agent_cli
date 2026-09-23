"""L1 编排 - 流式循环核心（从 orchestrator 提取）。"""

from .approval import (
    MAX_TOOL_FAILURES,
    extract_pending_tool_calls,
    handle_rejection,
    is_tool_error,
)
from .stream_helpers import (
    append_question_text_to_response,
    last_model_output_is_question,
)
from .types import AgentEvent, StepType


def run_stream_loop(orch, config: dict, pending_tool_calls: dict):
    """运行 agent 流式循环，yield 事件，返回循环状态 dict。

    负责：
    - 遍历 agent.stream 的 messages/updates 事件流
    - yield THINKING / TOKEN / THINK / ACT / APPROVE 事件
    - 处理中断 → 审批流程 → 拒绝注入
    - 连续失败计数与中止
    - 追问兜底
    """
    inputs = {"messages": orch.memory.get_messages()}
    respond_chunks: list[str] = []
    current_run_id: str | None = None
    thinking_sent_for_runs: set[str] = set()
    run_ids_with_tools: set[str] = set()
    consecutive_failures = 0
    aborted = False
    stream_input = inputs
    _memory_len_before = len(orch.memory.get_messages())

    while True:
        interrupted = False

        for chunk in orch.agent.stream(
            stream_input, config=config, stream_mode=["messages", "updates"]
        ):
            mode, data = chunk

            if mode == "updates" and "__interrupt__" in data:
                interrupted = True
                continue

            if mode == "messages":
                msg, metadata = data
                node = metadata.get("langgraph_node", "unknown")
                msg_type = type(msg).__name__
                run_id = getattr(msg, "id", "")

                if node == "model" and msg_type == "AIMessageChunk":
                    tool_calls = getattr(msg, "tool_calls", None)

                    if run_id and run_id != current_run_id:
                        current_run_id = run_id
                        if run_id not in thinking_sent_for_runs:
                            yield AgentEvent(step=StepType.THINKING, content="")
                            thinking_sent_for_runs.add(run_id)

                    if tool_calls:
                        run_ids_with_tools.add(run_id)
                        continue

                    content = getattr(msg, "content", "")
                    if content:
                        if run_id not in run_ids_with_tools:
                            yield AgentEvent(
                                step=StepType.TOKEN,
                                content=content,
                            )
                            respond_chunks.append(content)

            elif mode == "updates":
                for node_name, node_output in data.items():
                    for msg in node_output.get("messages", []):
                        if node_name == "model":
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                tool_details = []
                                for tc in tool_calls:
                                    name = tc["name"]
                                    args = tc.get("args", {})
                                    info = orch._tool_lookup.get(name)
                                    source = info.source if info else "unknown"
                                    category = info.category if info else ""
                                    pending_tool_calls[tc["id"]] = {
                                        "name": name,
                                        "source": source,
                                        "category": category,
                                        "args": args,
                                    }
                                    tool_details.append(
                                        {
                                            "name": name,
                                            "source": source,
                                            "category": category,
                                            "args": args,
                                        }
                                    )
                                yield AgentEvent(
                                    step=StepType.THINK,
                                    content="",
                                    metadata={"tool_details": tool_details},
                                )
                                orch.memory.add_message(msg)
                            else:
                                content = getattr(msg, "content", "")
                                orch.memory.add_ai(content)
                        elif node_name == "tools":
                            tc_id = getattr(msg, "tool_call_id", "")
                            info = pending_tool_calls.pop(tc_id, {})
                            tool_content = (
                                msg.content
                                if isinstance(msg.content, str)
                                else str(msg.content)
                            )

                            if is_tool_error(tool_content):
                                consecutive_failures += 1
                            else:
                                consecutive_failures = 0

                            yield AgentEvent(
                                step=StepType.ACT,
                                content=tool_content,
                                metadata={
                                    "tool_name": info.get("name", "unknown"),
                                    "tool_source": info.get("source", "unknown"),
                                    "tool_category": info.get("category", ""),
                                    "tool_args": info.get("args", {}),
                                    "tool_call_id": tc_id,
                                },
                            )
                            orch.memory.add_message(msg)

                            if consecutive_failures >= MAX_TOOL_FAILURES:
                                aborted = True
                                break

                    if aborted:
                        break

            if aborted:
                break

        if interrupted:
            if last_model_output_is_question(orch, config):
                append_question_text_to_response(orch, config, respond_chunks)
                break

            pending = extract_pending_tool_calls(orch)
            if pending:
                sensitive = [tc for tc in pending if tc.get("sensitive", False)]
                if not sensitive:
                    stream_input = None
                    continue

                yield AgentEvent(
                    step=StepType.APPROVE,
                    content="",
                    metadata={"tool_calls": pending},
                )

                approved_list = orch._approval_callback(pending)

                rejected_ids = [
                    tc["id"]
                    for tc, approved in zip(pending, approved_list)
                    if not approved
                ]

                if rejected_ids:
                    handle_rejection(orch, rejected_ids)
                    for tc in pending:
                        if tc["id"] in rejected_ids:
                            yield AgentEvent(
                                step=StepType.ACT,
                                content="用户拒绝了此工具调用。",
                                metadata={
                                    "tool_name": tc["name"],
                                    "tool_source": tc.get("source", "unknown"),
                                    "tool_category": tc.get("category", ""),
                                    "tool_args": tc.get("args", {}),
                                    "tool_call_id": tc["id"],
                                    "rejected": True,
                                },
                            )
                            consecutive_failures += 1

                    if consecutive_failures >= MAX_TOOL_FAILURES:
                        aborted = True
                        break

                stream_input = None
                continue
            else:
                stream_input = None
                continue

        break

    return {
        "respond_chunks": respond_chunks,
        "aborted": aborted,
        "consecutive_failures": consecutive_failures,
        "memory_len_before": _memory_len_before,
    }
