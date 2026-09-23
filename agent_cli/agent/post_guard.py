"""L1 编排 - 后置安全检测与响应最终化（从 orchestrator 提取）。"""

from .types import AgentEvent, StepType


def finalize_response(orch, user_input: str, state: dict):
    """组装最终响应，执行后置 guard，yield GUARD/RESPOND 事件。

    state 由 run_stream_loop 返回，包含：
    - respond_chunks: 流式累积的文本片段
    - aborted: 是否因连续失败中止
    - consecutive_failures: 连续失败次数
    - memory_len_before: 循环前消息数（用于 rollback）
    """
    full_response = "".join(state["respond_chunks"])
    if state["aborted"]:
        full_response = (
            f"工具连续调用失败 {state['consecutive_failures']} 次，已中止执行。"
            "请检查 MCP 工具配置或参数是否正确，然后重试。"
        )

    if orch._active_skill and orch._active_skill.postprocess:
        full_response = orch._active_skill.postprocess(full_response)

    post_guard = orch.middleware.after_run(user_input, full_response)
    if post_guard.findings:
        yield AgentEvent(
            step=StepType.GUARD,
            content=post_guard.message,
            metadata={
                "phase": "post",
                "risk_level": post_guard.risk_level,
                "blocked": post_guard.blocked,
                "findings": post_guard.findings,
            },
        )

    if post_guard.blocked:
        if len(orch.memory.get_messages()) > state["memory_len_before"]:
            orch.memory.rollback_to(state["memory_len_before"])
        full_response = f"[回复已拦截] 检测到安全风险：{post_guard.message}"
        orch.memory.add_ai(full_response)

    yield AgentEvent(
        step=StepType.RESPOND,
        content=full_response,
    )
