"""L1 编排核心 - Agent 循环主控。"""

import json
import uuid
from collections.abc import Iterator
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

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

VERIFY_SYSTEM_PROMPT = """\
你是一个任务验收助手。判断以下"用户目标"是否已被当前执行结果完全满足。

规则：
- 只输出一行，不要解释，不要 JSON。
- 如果目标已完全达成（结果可直接交付给用户），输出：[DONE]
- 如果还没有达成，需要 agent 继续，输出一行以 [CONTINUE] 开头，换行后给出：
  第一行写未达成的原因（简短），第二行写给 agent 的一条明确下一步指令。

注意：
- 不要为了追求完美而无限重做；一次 [CONTINUE] 只解决一个最关键的缺口。
- 如果目标是提问、闲聊或单纯聊天（没有要完成的具体事项），一律 [DONE]。
"""


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
        use_manual_react: bool = False,
        strategy: str | None = None,
        approval_mode: str | None = None,
        enable_verify: bool = False,
        verify_max_rounds: int = 2,
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
        if approval_callback is not None:
            self._approval_callback = approval_callback
            self._approval_mode = approval_mode or "full"
        else:
            self._approval_callback = builtin_approval_callback
            self._approval_mode = approval_mode or "sensitive"
        self._thread_id = "default"
        self._use_manual_react = use_manual_react
        self._react_max_steps = 8
        self._enable_verify = bool(enable_verify)
        self._verify_max_rounds = max(1, int(verify_max_rounds))
        valid_strategies = ("langgraph", "manual", "plan_execute")
        if strategy is not None and strategy not in valid_strategies:
            raise ValueError(f"未知策略: {strategy}，可选 {valid_strategies}")
        self._strategy = (
            strategy if strategy is not None else ("manual" if use_manual_react else "langgraph")
        )

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

    def set_model(self, model_name: str) -> Any:
        """动态切换主模型并重建 agent。

        切换会同步更新摘要压缩所用的模型。返回新的 model 实例。
        """
        self.model = self.provider.switch_model(model_name)
        if self.memory._summarizer is not None:
            from ..memory.summarizer import create_llm_summarizer

            self.memory._summarizer = create_llm_summarizer(self.model)
        self._rebind_memory_models()
        self._apply_skill()
        return self.model

    def set_provider(
        self, provider: BaseProvider, model_name: str | None = None
    ) -> Any:
        """整体切换 Provider（接口/密钥/模型名），并重建 agent 与记忆模型。"""
        if model_name:
            provider.switch_model(model_name)
        self.provider = provider
        self.model = provider.get_model()
        if self.memory._summarizer is not None:
            from ..memory.summarizer import create_llm_summarizer

            self.memory._summarizer = create_llm_summarizer(self.model)
        self._rebind_memory_models()
        self._apply_skill()
        return self.model

    def _rebind_memory_models(self) -> None:
        """切换模型/Provider 后，让长期记忆的 embedder 和事实提取器跟随新模型。

        重建失败时保持为空（对应功能自动降级），不中断切换流程。
        """
        try:
            get_embeddings = getattr(self.provider, "get_embeddings", None)
            self._embedder = get_embeddings() if get_embeddings else None
            if self._semantic_memory is not None:
                self._semantic_memory._embedder = self._embedder
                self._semantic_memory._embedder_failed = False

            from ..memory.extractor import create_fact_extractor

            self._fact_extractor = create_fact_extractor(self.model)
        except Exception:
            pass

    def get_active_skill(self) -> Skill | None:
        """获取当前激活的技能。"""
        return self._active_skill

    def set_approval_callback(
        self,
        callback: Callable[[list[dict]], list[bool]] | None,
        mode: str | None = None,
    ) -> None:
        """设置审批回调，并重建 agent。

        mode 取值：sensitive（仅敏感需确认）/ full（全部需确认）/ auto（全部自动通过）。
        不传 mode 时：None 回调 → 内置默认（sensitive）；自定义回调 → full。

        传入 None 时使用内置默认回调（自动批准非敏感工具，自动拒绝敏感工具）。
        """
        if callback is None:
            self._approval_callback = builtin_approval_callback
            self._approval_mode = "sensitive"
        else:
            self._approval_callback = callback
            self._approval_mode = mode or "full"
        self._apply_skill()

    def set_react_mode(self, enable: bool | None = None) -> bool:
        """切换 ReAct 循环实现：True=手动循环，False=LangGraph create_agent。

        返回切换后的模式。
        """
        if enable is not None:
            self._strategy = "manual" if enable else "langgraph"
            self._use_manual_react = bool(enable)
        return self._use_manual_react

    def set_strategy(self, name: str) -> str:
        """切换 Agent 循环策略：langgraph / manual / plan_execute。

        返回切换后的策略名。
        """
        valid_strategies = ("langgraph", "manual", "plan_execute")
        if name not in valid_strategies:
            raise ValueError(f"未知策略: {name}，可选 {valid_strategies}")
        self._strategy = name
        self._use_manual_react = name == "manual"
        return self._strategy

    def set_verify(self, enabled: bool | None = None, max_rounds: int | None = None) -> bool:
        """切换完成度评估闭环：True=执行后自动评估，未完成则继续迭代。

        返回切换后的开关状态。
        """
        if enabled is not None:
            self._enable_verify = bool(enabled)
        if max_rounds is not None:
            self._verify_max_rounds = max(1, int(max_rounds))
        return self._enable_verify

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

        if self._enable_verify:
            state = yield from self._run_verify_loop(processed_input, config, pending_tool_calls)
        else:
            state = yield from self._run_single_loop(config, pending_tool_calls)
        yield from finalize_response(self, user_input, state)

    def _run_single_loop(self, config: dict, pending_tool_calls: dict) -> Iterator[dict]:
        """按当前策略执行一轮 Agent 循环，返回与各 loop 兼容的 state dict。"""
        if self._strategy == "manual":
            from .strategies.react import manual_react_loop

            return (yield from manual_react_loop(self, config, pending_tool_calls))
        if self._strategy == "plan_execute":
            from .strategies.plan_execute import run_plan_execute_loop

            return (yield from run_plan_execute_loop(self, config, pending_tool_calls))
        return (yield from run_stream_loop(self, config, pending_tool_calls))

    def _run_verify_loop(
        self, user_input: str, config: dict, pending_tool_calls: dict
    ) -> Iterator[dict]:
        """完成度评估闭环：执行 → 评估 → 未完成则注入下一步指令重跑。

        独立于循环策略（manual / plan_execute / langgraph 均适用）。
        返回与 finalize_response 兼容的 state dict。
        """
        all_respond: list[str] = []
        aborted = False
        consecutive_failures = 0
        rounds = self._verify_max_rounds
        memory_len_before = len(self.memory.get_messages())

        for round_index in range(rounds):
            state = yield from self._run_single_loop(config, pending_tool_calls)
            all_respond.extend(state.get("respond_chunks", []) or [])
            aborted = bool(state.get("aborted"))
            consecutive_failures = state.get("consecutive_failures", 0)

            if aborted:
                break
            if round_index == rounds - 1:
                break

            verdict = self._evaluate_verdict(user_input, all_respond)
            if verdict is None or not verdict.get("continue"):
                break
            next_action = str(verdict.get("next_action", "")).strip()
            if not next_action:
                break
            reason = str(verdict.get("reason", ""))
            yield AgentEvent(
                step=StepType.THINK,
                content=next_action,
                metadata={"phase": "verify", "reason": reason},
            )
            self.memory.add_human(next_action)

        return {
            "respond_chunks": all_respond,
            "aborted": aborted,
            "consecutive_failures": consecutive_failures,
            "memory_len_before": memory_len_before,
        }

    def _evaluate_verdict(self, user_input: str, respond_chunks: list[str]) -> dict | None:
        """让模型判断任务是否完成。

        返回 {"continue": bool, "reason": str, "next_action": str}；
        模型不可用/无法解析时返回 None（视为已完成，安全默认）。
        """
        respond_text = "".join(respond_chunks)[-2000:] if respond_chunks else "(尚无回复)"
        messages = [
            SystemMessage(content=VERIFY_SYSTEM_PROMPT),
            HumanMessage(content=f"用户目标：{user_input}"),
            HumanMessage(content=f"当前执行结果：\n{respond_text}"),
        ]
        try:
            resp = self.model.invoke(messages)
        except Exception:
            return None
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip()
        if text.startswith("[DONE]"):
            return {
                "continue": False,
                "reason": text[len("[DONE]"):].strip(),
                "next_action": "",
            }
        if text.startswith("[CONTINUE]"):
            after = text[len("[CONTINUE]"):].strip()
            reason, _, action = after.partition("\n")
            return {
                "continue": True,
                "reason": reason.strip(),
                "next_action": action.strip() or reason.strip(),
            }
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("done"), bool):
                return {
                    "continue": not data["done"],
                    "reason": str(data.get("reason", "") or ""),
                    "next_action": str(data.get("next_action", "") or ""),
                }
        except Exception:
            pass
        return None

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
