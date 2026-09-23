"""L1 编排核心 - Agent 循环主控。"""

import uuid
from collections.abc import Iterator
from typing import Any, Callable

from langchain.agents import create_agent

from ..mcp.registry import MCPRegistry
from ..memory.summarizer import create_llm_summarizer
from ..memory.working import WorkingMemory
from ..middleware.pipeline import MiddlewarePipeline, create_default_pipeline
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT
from ..providers.base import BaseProvider
from ..skills.define_skill import Skill
from ..tools.registry import ToolInfo, create_default_registry
from .approval import builtin_approval_callback
from .config_info import build_config_info
from .long_term_memory import (
    init_long_term_memory,
    inject_episodic_memory,
    inject_semantic_memory,
    save_session,
)
from .post_guard import finalize_response
from .stream_runner import run_stream_loop
from .types import AgentEvent, StepType


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
        approval_callback: Callable[[list[dict]], list[bool]] | None = None,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        compression_threshold: int | None = None,
        keep_recent: int | None = None,
        summarizer_model: Any = None,
        enable_compression: bool = True,
        enable_long_term_memory: bool = False,
        db_path: str | None = None,
    ):
        self.provider = provider
        self.model = provider.get_model()
        self.memory = memory or WorkingMemory()
        if max_messages is not None:
            self.memory.max_messages = max_messages

        if enable_compression:
            summarizer_model = summarizer_model or self.model
            self.memory._summarizer = create_llm_summarizer(summarizer_model)
            if max_tokens is not None:
                self.memory._max_tokens = max_tokens
            if compression_threshold is not None:
                self.memory._compression_threshold = compression_threshold
            if keep_recent is not None:
                self.memory._keep_recent = keep_recent
        else:
            self.memory._summarizer = None
        self.middleware = middleware or create_default_pipeline()
        self._mcp_registry = mcp_registry
        self._active_skill = skill
        self._base_system_prompt = system_prompt
        self._approval_callback = (
            approval_callback
            if approval_callback is not None
            else builtin_approval_callback
        )
        self._thread_id = "default"

        self.registry = create_default_registry()

        if extra_tool_infos:
            for info in extra_tool_infos:
                self.registry.register(
                    info.tool, source=info.source, category=info.category
                )

        if mcp_registry:
            for info in mcp_registry.get_tool_infos():
                self.registry.register(
                    info.tool, source=info.source, category=info.category
                )

        self._all_tools_info = self.registry.get_all_info()
        self._tool_lookup = {info.name: info for info in self._all_tools_info}

        self._ltm_enabled = enable_long_term_memory
        self._db_conn = None
        self._episodic_memory = None
        self._semantic_memory = None
        self._fact_extractor = None
        self._embedder = None
        self._session_id = str(uuid.uuid4())
        self._semantic_injected = False

        if enable_long_term_memory:
            init_long_term_memory(self, db_path, summarizer_model or self.model)

        self._apply_skill()

    def _apply_skill(self) -> None:
        """根据当前 skill 过滤工具和设置 prompt，并重建 agent。"""
        if self._active_skill:
            self.system_prompt = self._active_skill.system_prompt
            if self._active_skill.tool_allowlist is not None:
                allowed = set(self._active_skill.tool_allowlist)
                filtered = [
                    info for info in self._all_tools_info if info.name in allowed
                ]
                self.tools = [info.tool for info in filtered]
                self._tool_lookup = {info.name: info for info in filtered}
            else:
                self.tools = self.registry.get_all()
                self._tool_lookup = {info.name: info for info in self._all_tools_info}
        else:
            self.system_prompt = self._base_system_prompt
            self.tools = self.registry.get_all()
            self._tool_lookup = {info.name: info for info in self._all_tools_info}

        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        interrupt_before = ["tools"]

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
        )

    def set_skill(self, skill: Skill | None) -> None:
        """动态切换技能。

        切换后会清空记忆，因为上下文可能不适用于新的 skill。
        """
        self._active_skill = skill
        self.memory.clear()
        if self._ltm_enabled:
            inject_episodic_memory(self)
            self._semantic_injected = False
        self._apply_skill()

    def get_active_skill(self) -> Skill | None:
        """获取当前激活的技能。"""
        return self._active_skill

    def set_approval_callback(
        self, callback: Callable[[list[dict]], list[bool]] | None
    ) -> None:
        """设置审批回调，并重建 agent。

        传入 None 时使用内置默认回调（自动批准非敏感工具，自动拒绝敏感工具）。
        """
        self._approval_callback = (
            callback if callback is not None else builtin_approval_callback
        )
        self._apply_skill()

    def run_stream(self, user_input: str) -> Iterator[AgentEvent]:
        """逐步 yield guard/thinking/think/act/token/respond/approve 事件。

        事件顺序：
        0. GUARD     — 前置安全检测（拦截或警告）
        1. THINKING  — 模型开始推理（spinner）
        2. THINK     — 模型决定调用工具（工具名/参数/标签）
        3. APPROVE   — 工具调用前等待审批（仅启用审批时）
        4. ACT       — 工具执行结果
        5. TOKEN     — 最终回复的逐 token 流（打字机效果）
        6. RESPOND   — 最终回复结束
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

        processed_input = user_input
        if self._active_skill and self._active_skill.preprocess:
            processed_input = self._active_skill.preprocess(user_input)

        if self._ltm_enabled and not self._semantic_injected:
            inject_semantic_memory(self, processed_input)
            self._semantic_injected = True

        self.memory.add_human(processed_input)
        self._thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": self._thread_id}}
        pending_tool_calls: dict[str, dict] = {}

        state = yield from run_stream_loop(self, config, pending_tool_calls)
        yield from finalize_response(self, user_input, state)

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
        if self._ltm_enabled:
            inject_episodic_memory(self)
            self._semantic_injected = False

    def get_config_info(self) -> dict[str, Any]:
        """返回当前 Agent 配置的摘要信息。"""
        return build_config_info(self)

    def cleanup(self) -> None:
        """关闭外部连接（MCP server 等），持久化长期记忆。"""
        if self._ltm_enabled:
            try:
                save_session(self)
            except Exception:
                pass
            finally:
                if self._db_conn:
                    self._db_conn.close()

        if self._mcp_registry:
            self._mcp_registry.disconnect_all()
