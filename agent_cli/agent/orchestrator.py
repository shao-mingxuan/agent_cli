"""L1 编排核心 - Agent 循环主控。"""
import uuid
from collections.abc import Iterator
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from ..mcp.registry import MCPRegistry
from ..memory.summarizer import create_llm_summarizer
from ..memory.working import WorkingMemory
from ..middleware.pipeline import MiddlewarePipeline, create_default_pipeline
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT
from ..providers.base import BaseProvider
from ..skills.define_skill import Skill
from ..tools.registry import ToolInfo, create_default_registry
from .types import AgentEvent, StepType

MAX_TOOL_FAILURES = 3
_TOOL_ERROR_MARKERS = ("error", "mcp error", "exception", "traceback", "failed")

# 敏感工具关键词：匹配工具名或分类，命中则触发人工审批
# 敏感工具关键词 —— 按匹配策略分为两类：
#
# 1. _SENSITIVE_WORDS：按单词匹配（以下划线或空白分隔）。
#    避免短关键词在子串中误匹配（如 "sh" 命中 "push"、"db" 命中 "adblock"）。
# 2. _SENSITIVE_PATTERNS：按子串匹配，仅包含长度 >=5 的特定完整词组，
#    因长度足够，几乎不会产生误匹配。
_SENSITIVE_WORDS: set[str] = {
    # 执行类
    "shell", "exec", "bash", "spawn", "command", "run",
    # 数据库类
    "sql", "database", "db",
    "insert", "update", "drop",
    # 文件系统 — 写/删/改
    "write", "delete", "remove", "create", "mkdir", "rmdir",
    "edit", "modify", "patch",
    "move", "copy", "rename", "chmod", "chown",
    # 文件/目录 — 读操作也视为敏感（可读取密钥等）
    "file", "directory",
    # 版本控制
    "git", "commit", "push", "pull",
    # 其它
    "fs",
}

_SENSITIVE_PATTERNS: set[str] = {
    "filesystem",
    "read_file", "write_file", "edit_file",
    "read_directory", "list_directory", "search_files",
    "fs_",
}


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
            else self._builtin_approval_callback
        )
        self._thread_id = "default"

        self.registry = create_default_registry()

        if extra_tool_infos:
            for info in extra_tool_infos:
                self.registry.register(info.tool, source=info.source, category=info.category)

        if mcp_registry:
            for info in mcp_registry.get_tool_infos():
                self.registry.register(info.tool, source=info.source, category=info.category)

        self._all_tools_info = self.registry.get_all_info()
        self._tool_lookup = {info.name: info for info in self._all_tools_info}

        # 长期记忆
        self._ltm_enabled = enable_long_term_memory
        self._db_conn = None
        self._episodic_memory = None
        self._semantic_memory = None
        self._fact_extractor = None
        self._embedder = None
        self._session_id = str(uuid.uuid4())
        self._semantic_injected = False

        if enable_long_term_memory:
            self._init_long_term_memory(
                db_path, summarizer_model or self.model
            )

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

    def _init_long_term_memory(
        self, db_path: str | None, extractor_model: Any
    ) -> None:
        """初始化长期记忆：SQLite 连接、episodic/semantic 记忆、事实提取器。"""
        from ..memory.episodic import EpisodicMemory
        from ..memory.extractor import create_fact_extractor
        from ..memory.semantic import SemanticMemory
        from ..memory.store import get_connection

        self._db_conn = get_connection(db_path)

        get_embeddings = getattr(self.provider, "get_embeddings", None)
        if get_embeddings:
            try:
                self._embedder = get_embeddings()
            except Exception:
                self._embedder = None

        self._episodic_memory = EpisodicMemory(self._db_conn)
        self._semantic_memory = SemanticMemory(
            self._db_conn, embedder=self._embedder
        )

        try:
            self._fact_extractor = create_fact_extractor(extractor_model)
        except Exception:
            self._fact_extractor = None

        self._inject_episodic_memory()

    def _inject_episodic_memory(self) -> None:
        """注入历史会话回忆到 WorkingMemory。"""
        if self._episodic_memory is None:
            return
        try:
            text = self._episodic_memory.format_recall()
            if text:
                self.memory.add_message(SystemMessage(content=text))
        except Exception:
            pass

    def _inject_semantic_memory(self, user_input: str) -> None:
        """首次输入时检索语义记忆并注入到 WorkingMemory。"""
        if self._semantic_memory is None:
            return
        try:
            text = self._semantic_memory.format_recall(user_input)
            if text:
                self.memory.add_message(SystemMessage(content=text))
        except Exception:
            pass

    def _save_session(self) -> None:
        """提取会话摘要和事实，持久化到 SQLite。"""
        if self._episodic_memory is None:
            return

        messages = self.memory.get_messages()
        if len(messages) < 4:
            return

        text = self.memory._serialize_messages(messages)

        if self.memory._summarizer:
            try:
                summary = self.memory._summarizer(text)
                if summary:
                    self._episodic_memory.save_session(
                        self._session_id, summary
                    )
            except Exception:
                pass

        if self._fact_extractor:
            try:
                facts = self._fact_extractor(text)
                for fact in facts:
                    emb = None
                    if self._embedder:
                        try:
                            emb = self._embedder(fact)
                        except Exception:
                            pass
                    self._semantic_memory.store_fact(
                        fact, source=self._session_id, embedding=emb
                    )
            except Exception:
                pass

    def set_skill(self, skill: Skill | None) -> None:
        """动态切换技能。

        切换后会清空记忆，因为上下文可能不适用于新的 skill。
        """
        self._active_skill = skill
        self.memory.clear()
        if self._ltm_enabled:
            self._inject_episodic_memory()
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
            callback
            if callback is not None
            else self._builtin_approval_callback
        )
        self._apply_skill()

    def _is_sensitive_tool(self, tool_call: dict) -> bool:
        """判断工具调用是否为敏感操作（需要人工审批）。

        匹配策略：
        1. 单词匹配（以下划线或空白分隔）——防止短关键词在子串中误匹配
           例如 "sh" 不再命中 "push"，"db" 不再命中 "adblock"。
        2. 子串匹配 —— 仅对长度 >=5 的特定完整词组生效。
        """
        name = tool_call.get("name", "").lower()
        category = tool_call.get("category", "").lower()
        combined = f"{name} {category}"

        # 1. 单词匹配（按 _ 或空白切分）
        words = set(combined.replace("_", " ").split())
        if words & _SENSITIVE_WORDS:
            return True

        # 2. 子串匹配（仅长词组，避免误匹配）
        for pattern in _SENSITIVE_PATTERNS:
            if pattern in combined:
                return True

        return False

    @staticmethod
    def _builtin_approval_callback(tool_calls: list[dict]) -> list[bool]:
        """内置默认审批回调：非敏感自动通过，敏感自动拒绝（安全默认）。"""
        return [not tc.get("sensitive", False) for tc in tool_calls]

    def _extract_pending_tool_calls(self) -> list[dict]:
        """从中断状态中提取待审批的工具调用。"""
        config = {"configurable": {"thread_id": self._thread_id}}
        state = self.agent.get_state(config)
        messages = state.values.get("messages", [])
        pending = []
        for msg in reversed(messages):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc.get("args", {})
                    info = self._tool_lookup.get(name)
                    pending.append(
                        {
                            "id": tc["id"],
                            "name": name,
                            "args": args,
                            "source": info.source if info else "unknown",
                            "category": info.category if info else "",
                            "sensitive": self._is_sensitive_tool(
                                {
                                    "name": name,
                                    "category": info.category if info else "",
                                }
                            ),
                        }
                    )
                break
        return pending

    def _last_model_output_is_question(self, config: dict) -> bool:
        """检查对话中最后一条 AIMessage 的纯文本是否以问号结尾。

        用于追问兜底：如果模型正在向用户澄清（以问号结尾的追问），
        同一回合不应再继续执行任何工具调用。
        """
        state = self.agent.get_state(config)
        messages = state.values.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    return content.strip().endswith(("?", "？"))
                break
        return False

    def _append_question_text_to_response(
        self, config: dict, respond_chunks: list[str]
    ) -> None:
        """把模型追问文本追加到 respond_chunks。

        追问兜底触发时，模型可能在同一个 AIMessage 里既输出追问文本又发起了
        工具调用，此时文本已被跳过未累积。这里从 checkpoint 中找回它，
        确保 break 后追问内容能作为最终回复输出给用户。
        """
        state = self.agent.get_state(config)
        messages = state.values.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    respond_chunks.append(content)
                break

    def _handle_rejection(self, rejected_ids: list[str]) -> None:
        """对被拒绝的工具调用注入拒绝 ToolMessage 并更新状态。"""
        config = {"configurable": {"thread_id": self._thread_id}}
        rejected_msgs = [
            ToolMessage(
                content="用户拒绝了此工具调用。",
                name="rejected",
                tool_call_id=tc_id,
            )
            for tc_id in rejected_ids
        ]
        if rejected_msgs:
            self.agent.update_state(
                config, {"messages": rejected_msgs}, as_node="tools"
            )

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

        # 应用 skill 预处理
        processed_input = user_input
        if self._active_skill and self._active_skill.preprocess:
            processed_input = self._active_skill.preprocess(user_input)

        # 长期记忆：首次输入时注入语义回忆
        if self._ltm_enabled and not self._semantic_injected:
            self._inject_semantic_memory(processed_input)
            self._semantic_injected = True

        self.memory.add_human(processed_input)
        self._thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": self._thread_id}}
        pending_tool_calls: dict[str, dict] = {}

        yield from self._run_stream_with_approval(
            user_input, config, pending_tool_calls
        )

    def _run_stream_with_approval(
        self,
        user_input: str,
        config: dict,
        pending_tool_calls: dict[str, dict],
    ) -> Iterator[AgentEvent]:
        """带审批模式的流式执行：工具调用前暂停等用户确认。"""
        inputs = {"messages": self.memory.get_messages()}
        respond_chunks: list[str] = []
        current_run_id: str | None = None
        thinking_sent_for_runs: set[str] = set()
        run_ids_with_tools: set[str] = set()
        consecutive_failures = 0
        aborted = False
        stream_input = inputs
        # TOCTOU 修复：记录本轮开始前的 memory 长度，blocked 时回滚
        _memory_len_before = len(self.memory.get_messages())

        while True:
            interrupted = False

            for chunk in self.agent.stream(
                stream_input, config=config, stream_mode=["messages", "updates"]
            ):
                mode, data = chunk

                # 检测中断
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
                                yield AgentEvent(
                                    step=StepType.THINKING, content=""
                                )
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
                                        info = self._tool_lookup.get(name)
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
                                        metadata={
                                            "tool_details": tool_details
                                        },
                                    )
                                    self.memory.add_message(msg)
                                else:
                                    content = getattr(msg, "content", "")
                                    self.memory.add_ai(content)
                            elif node_name == "tools":
                                tc_id = getattr(msg, "tool_call_id", "")
                                info = pending_tool_calls.pop(tc_id, {})
                                tool_content = (
                                    msg.content
                                    if isinstance(msg.content, str)
                                    else str(msg.content)
                                )

                                if _is_tool_error(tool_content):
                                    consecutive_failures += 1
                                else:
                                    consecutive_failures = 0

                                yield AgentEvent(
                                    step=StepType.ACT,
                                    content=tool_content,
                                    metadata={
                                        "tool_name": info.get("name", "unknown"),
                                        "tool_source": info.get(
                                            "source", "unknown"
                                        ),
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

                if aborted:
                    break

            if interrupted:
                # 追问兜底：读取模型本次完整输出，若以问号结尾（正在向用户澄清），
                # 同一回合立即停止，不再执行任何工具，等待用户回复后再继续。
                if self._last_model_output_is_question(config):
                    self._append_question_text_to_response(config, respond_chunks)
                    break

                # 提取待审批工具调用
                pending = self._extract_pending_tool_calls()
                if pending:
                    # 只拦截敏感工具；非敏感工具自动放行
                    sensitive = [tc for tc in pending if tc.get("sensitive", False)]
                    if not sensitive:
                        stream_input = None
                        continue

                    yield AgentEvent(
                        step=StepType.APPROVE,
                        content="",
                        metadata={"tool_calls": pending},
                    )

                    # 调用审批回调
                    approved_list = self._approval_callback(pending)

                    # 分离批准和拒绝
                    rejected_ids = [
                        tc["id"]
                        for tc, approved in zip(pending, approved_list)
                        if not approved
                    ]
                    approved_ids = [
                        tc["id"]
                        for tc, approved in zip(pending, approved_list)
                        if approved
                    ]

                    # 如果有被拒绝的工具，注入拒绝消息
                    if rejected_ids:
                        self._handle_rejection(rejected_ids)
                        # yield 拒绝的 ACT 事件
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

                    # 恢复执行（None 表示从断点继续）
                    stream_input = None
                    continue
                else:
                    # 没有待审批的工具，直接恢复
                    stream_input = None
                    continue

            # 没有中断，退出循环
            break

        # 后置处理
        full_response = "".join(respond_chunks)
        if aborted:
            full_response = (
                f"工具连续调用失败 {consecutive_failures} 次，已中止执行。"
                "请检查 MCP 工具配置或参数是否正确，然后重试。"
            )

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
            # TOCTOU 修复：回滚本轮写入 memory 的所有消息，避免不安全内容残留
            if len(self.memory.get_messages()) > _memory_len_before:
                self.memory.rollback_to(_memory_len_before)
            full_response = f"[回复已拦截] 检测到安全风险：{post_guard.message}"
            self.memory.add_ai(full_response)

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
        if self._ltm_enabled:
            self._inject_episodic_memory()
            self._semantic_injected = False

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
                mcp_tools.append(
                    {
                        "name": info.name,
                        "category": info.category,
                    }
                )

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
            "approval_mode": (
                "full"
                if self._approval_callback is not self._builtin_approval_callback
                else "sensitive"
            ),
            "max_messages": self.memory.max_messages,
            "max_tokens": self.memory.max_tokens,
            "compression_enabled": self.memory.compression_enabled,
            "long_term_memory": self._ltm_enabled,
            "episodic_count": (
                self._episodic_memory.count()
                if self._episodic_memory
                else 0
            ),
            "semantic_count": (
                self._semantic_memory.count()
                if self._semantic_memory
                else 0
            ),
        }

    def cleanup(self) -> None:
        """关闭外部连接（MCP server 等），持久化长期记忆。"""
        if self._ltm_enabled:
            try:
                self._save_session()
            except Exception:
                pass
            finally:
                if self._db_conn:
                    self._db_conn.close()

        if self._mcp_registry:
            self._mcp_registry.disconnect_all()
