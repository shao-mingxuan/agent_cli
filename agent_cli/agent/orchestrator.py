"""L1 编排核心 - Agent 循环主控。"""
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

from ..memory.working import WorkingMemory
from ..providers.base import BaseProvider
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT
from ..tools.registry import ToolRegistry, create_default_registry
from ..middleware.pipeline import MiddlewarePipeline, create_default_pipeline
from ..middleware.types import GuardResult
from .types import AgentState, AgentEvent, StepType


class Orchestrator:
    """Agent 编排主控：组装 model + tools + memory，运行 agent 循环。"""

    def __init__(
        self,
        provider: BaseProvider,
        tools: list[Any] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: WorkingMemory | None = None,
        middleware: MiddlewarePipeline | None = None,
    ):
        self.provider = provider
        self.model = provider.get_model()
        self.memory = memory or WorkingMemory()
        self.system_prompt = system_prompt
        self.middleware = middleware or create_default_pipeline()

        if tools is None:
            self.registry = create_default_registry()
            tools = self.registry.get_all()
        else:
            self.registry = ToolRegistry()
            for t in tools:
                self.registry.register(t, source="unknown")

        self.tools = tools
        self._tool_lookup = {
            info.name: info for info in self.registry.get_all_info()
        }
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def run_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """逐步 yield guard/thinking/think/act/token/respond 事件。

        事件顺序：
        0. GUARD     — 前置安全检测（拦截或警告）
        1. THINKING  — 模型开始推理（spinner）
        2. THINK     — 模型决定调用工具（工具名/参数/标签）
        3. ACT       — 工具执行结果
        4. TOKEN     — 最终回复的逐 token 流（打字机效果）
        5. RESPOND   — 最终回复结束
        """
        pre_guard = self.middleware.before_run(user_input)
        yield AgentEvent(
            step=StepType.GUARD,
            content=pre_guard.message,
            metadata={
                "phase": "pre",
                "risk_level": pre_guard.risk_level,
                "blocked": pre_guard.blocked,
                "findings": pre_guard.findings,
            },
        )

        if pre_guard.blocked:
            yield AgentEvent(
                step=StepType.RESPOND,
                content=f"[已拦截] 输入存在安全风险：{pre_guard.message}",
            )
            return

        self.memory.add_human(user_input)
        inputs = {"messages": self.memory.get_messages()}
        pending_tool_calls: dict[str, dict] = {}
        # 跟踪当前模型调用的 run_id，用于区分不同轮次
        current_run_id: str | None = None
        # 记录是否已经为当前 run_id 发出过 thinking 事件
        thinking_sent_for_runs: set[str] = set()
        # 记录哪些 run_id 是有 tool_calls 的（非最终回复）
        run_ids_with_tools: set[str] = set()
        # 拼接最终回复
        respond_chunks: list[str] = []
        # 跟踪是否已经发出了 THINK 事件
        think_emitted = False

        for chunk in self.agent.stream(inputs, stream_mode=["messages", "updates"]):
            mode, data = chunk

            if mode == "messages":
                msg, metadata = data
                node = metadata.get("langgraph_node", "unknown")
                msg_type = type(msg).__name__
                run_id = getattr(msg, "id", "")

                if node == "model" and msg_type == "AIMessageChunk":
                    tool_calls = getattr(msg, "tool_calls", None)

                    # 新的模型调用轮次开始
                    if run_id and run_id != current_run_id:
                        current_run_id = run_id
                        think_emitted = False
                        # 如果还没为这个 run_id 发过 thinking，发一次
                        if run_id not in thinking_sent_for_runs:
                            yield AgentEvent(step=StepType.THINKING, content="")
                            thinking_sent_for_runs.add(run_id)

                    if tool_calls:
                        run_ids_with_tools.add(run_id)
                        # tool_calls 在 chunk 中累积，updates 模式会给出完整 AIMessage
                        # 这里不 yield，等 updates 模式的完整消息
                        continue

                    # 有 content 的 token
                    content = getattr(msg, "content", "")
                    if content:
                        # 如果这个 run_id 不在 tools 集合中，说明是最终回复的 token
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
                                    info = self._tool_lookup.get(name)
                                    source = info.source if info else "unknown"
                                    category = info.category if info else ""
                                    pending_tool_calls[tc["id"]] = {
                                        "name": name,
                                        "source": source,
                                        "category": category,
                                        "args": args,
                                    }
                                    tool_details.append({
                                        "name": name,
                                        "source": source,
                                        "category": category,
                                        "args": args,
                                    })
                                yield AgentEvent(
                                    step=StepType.THINK,
                                    content="",
                                    metadata={"tool_details": tool_details},
                                )
                                think_emitted = True
                                self.memory.add_message(msg)
                            else:
                                content = getattr(msg, "content", "")
                                self.memory.add_ai(content)
                        elif node_name == "tools":
                            tc_id = getattr(msg, "tool_call_id", "")
                            info = pending_tool_calls.pop(tc_id, {})
                            yield AgentEvent(
                                step=StepType.ACT,
                                content=msg.content,
                                metadata={
                                    "tool_name": info.get("name", "unknown"),
                                    "tool_source": info.get("source", "unknown"),
                                    "tool_category": info.get("category", ""),
                                    "tool_args": info.get("args", {}),
                                    "tool_call_id": tc_id,
                                },
                            )
                            self.memory.add_message(msg)

        # 后置安全检测
        full_response = "".join(respond_chunks)
        post_guard = self.middleware.after_run(user_input, full_response)
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
            full_response = f"[回复已拦截] 检测到安全风险：{post_guard.message}"

        yield AgentEvent(
            step=StepType.RESPOND,
            content=full_response,
        )

    def run(self, user_input: str) -> AgentEvent:
        """非流式执行，返回最终 respond 事件。"""
        final = None
        for event in self.run_stream(user_input):
            if event.step == StepType.RESPOND:
                final = event
        return final or AgentEvent(step=StepType.RESPOND, content="无回复")

    def reset(self) -> None:
        """清空对话记忆。"""
        self.memory.clear()
