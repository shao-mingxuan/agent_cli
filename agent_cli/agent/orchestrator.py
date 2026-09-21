"""L1 编排核心 - Agent 循环主控。"""
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

from ..memory.working import WorkingMemory
from ..providers.base import BaseProvider
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT
from ..tools.registry import ToolRegistry, create_default_registry, ToolInfo
from ..middleware.pipeline import MiddlewarePipeline, create_default_pipeline
from ..middleware.types import GuardResult
from ..mcp.registry import MCPRegistry
from ..skills.define_skill import Skill
from .types import AgentState, AgentEvent, StepType

MAX_TOOL_FAILURES = 3
_TOOL_ERROR_MARKERS = ("error", "mcp error", "exception", "traceback", "failed")


def _is_tool_error(content: str) -> bool:
    """检测工具返回内容是否为错误。"""
    if not content:
        return False
    lower = content.lower()
    return any(marker in lower for marker in _TOOL_ERROR_MARKERS)


class Orchestrator:
    """Agent 编排主控：组装 model + tools + memory，运行 agent 循环。"""

    def __init__(
        self,
        provider: BaseProvider,
        tools: list[Any] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: WorkingMemory | None = None,
        middleware: MiddlewarePipeline | None = None,
        mcp_registry: MCPRegistry | None = None,
        extra_tool_infos: list[ToolInfo] | None = None,
        skill: Skill | None = None,
    ):
        self.provider = provider
        self.model = provider.get_model()
        self.memory = memory or WorkingMemory()
        self.middleware = middleware or create_default_pipeline()
        self._mcp_registry = mcp_registry
        self._active_skill = skill
        self._base_system_prompt = system_prompt

        self.registry = create_default_registry()

        if extra_tool_infos:
            for info in extra_tool_infos:
                self.registry.register(info.tool, source=info.source, category=info.category)

        if mcp_registry:
            for info in mcp_registry.get_tool_infos():
                self.registry.register(info.tool, source=info.source, category=info.category)

        self._all_tools_info = self.registry.get_all_info()
        self._tool_lookup = {info.name: info for info in self._all_tools_info}

        # 应用 skill 配置并创建 agent
        self._apply_skill()

    def _apply_skill(self) -> None:
        """根据当前 skill 过滤工具和设置 prompt，并重建 agent。"""
        if self._active_skill:
            self.system_prompt = self._active_skill.system_prompt
            if self._active_skill.tool_allowlist is not None:
                allowed = set(self._active_skill.tool_allowlist)
                filtered = [info for info in self._all_tools_info if info.name in allowed]
                self.tools = [info.tool for info in filtered]
                self._tool_lookup = {info.name: info for info in filtered}
            else:
                self.tools = self.registry.get_all()
                self._tool_lookup = {info.name: info for info in self._all_tools_info}
        else:
            self.system_prompt = self._base_system_prompt
            self.tools = self.registry.get_all()
            self._tool_lookup = {info.name: info for info in self._all_tools_info}

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def set_skill(self, skill: Skill | None) -> None:
        """动态切换技能。

        切换后会清空记忆，因为上下文可能不适用于新的 skill。
        """
        self._active_skill = skill
        self.memory.clear()
        self._apply_skill()

    def get_active_skill(self) -> Skill | None:
        """获取当前激活的技能。"""
        return self._active_skill

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

        # 应用 skill 预处理
        processed_input = user_input
        if self._active_skill and self._active_skill.preprocess:
            processed_input = self._active_skill.preprocess(user_input)

        self.memory.add_human(processed_input)
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
        # 工具连续失败计数
        consecutive_failures = 0
        # 是否因失败过多而中止
        aborted = False

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
                            tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)

                            if _is_tool_error(tool_content):
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
                            self.memory.add_message(msg)

                            if consecutive_failures >= MAX_TOOL_FAILURES:
                                aborted = True
                                break

                    if aborted:
                        break

        # 后置安全检测
        full_response = "".join(respond_chunks)

        # 工具连续失败中止
        if aborted:
            full_response = (
                f"工具连续调用失败 {consecutive_failures} 次，已中止执行。"
                "请检查 MCP 工具配置或参数是否正确，然后重试。"
            )

        # 应用 skill 后处理
        if self._active_skill and self._active_skill.postprocess:
            full_response = self._active_skill.postprocess(full_response)

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

    def get_config_info(self) -> dict[str, Any]:
        """返回当前 Agent 配置的摘要信息。"""
        model = getattr(self.provider, "model_name", "unknown")
        base_url = getattr(self.provider, "base_url", "unknown")

        tools_by_source: dict[str, list[str]] = {}
        for name, info in self._tool_lookup.items():
            tools_by_source.setdefault(info.source, []).append(name)

        mcp_tools: list[dict] = []
        if self._mcp_registry:
            for info in self._mcp_registry.get_tool_infos():
                mcp_tools.append({
                    "name": info.name,
                    "category": info.category,
                })

        skill_info = None
        if self._active_skill:
            skill_info = {
                "name": self._active_skill.name,
                "description": self._active_skill.description,
                "tool_allowlist": self._active_skill.tool_allowlist,
            }

        return {
            "model": model,
            "base_url": base_url,
            "system_prompt_preview": self.system_prompt[:200]
            + ("..." if len(self.system_prompt) > 200 else ""),
            "total_tools": len(self._tool_lookup),
            "tools_by_source": tools_by_source,
            "mcp_tools": mcp_tools,
            "skill": skill_info,
        }

    def cleanup(self) -> None:
        """关闭外部连接（MCP server 等）。"""
        if self._mcp_registry:
            self._mcp_registry.disconnect_all()
