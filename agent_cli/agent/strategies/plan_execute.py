"""L1 编排核心 - Plan-Execute 策略。"""

from typing import Any, Iterator

from langchain_core.messages import SystemMessage

from ..types import AgentEvent, StepType
from .react import manual_react_loop

PLAN_SYSTEM_PROMPT = """\
你是一个擅长规划的助手。在动手前，先为用户的请求输出一份简洁的分步计划。

要求：
- 用编号列表输出，每步一句清晰描述
- 若任务需要读取文件/运行命令/查数据，计划中要明确提到对应工具
- 计划不超过 8 步
- 只输出计划本身，不要先执行任何工具
"""


def _to_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def run_plan_execute_loop(
    orch, config: dict, pending_tool_calls: dict
) -> Iterator[dict]:
    """Plan-Execute 循环：先生成计划，再按计划执行。

    阶段一：用不带工具的模型调用生成计划，作为 THINK 事件展示给用户。
    阶段二：委托 manual_react_loop 执行循环（工具调用/审批/失败计数一致）。

    返回 state dict 与 run_stream_loop / manual_react_loop 兼容。
    """
    respond_chunks: list[str] = []
    aborted = False
    _memory_len_before = len(orch.memory.get_messages())

    messages = [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        SystemMessage(content=orch.system_prompt),
    ] + orch.memory.get_messages()

    try:
        plan_msg = orch.model.invoke(messages)
    except Exception as e:
        error_text = f"[模型调用失败] {type(e).__name__}: {e}"
        respond_chunks.append(error_text)
        yield AgentEvent(step=StepType.RESPOND, content=error_text)
        return {
            "respond_chunks": respond_chunks,
            "aborted": True,
            "consecutive_failures": 0,
            "memory_len_before": _memory_len_before,
        }

    plan_text = _to_text(plan_msg.content).strip()
    if plan_text:
        yield AgentEvent(
            step=StepType.THINK,
            content=plan_text,
            metadata={"plan": plan_text},
        )
        orch.memory.add_message(plan_msg)

    state = yield from manual_react_loop(orch, config, pending_tool_calls)
    state["memory_len_before"] = _memory_len_before
    return state
