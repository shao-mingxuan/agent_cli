"""L0 CLI - 终端展示。"""

import shutil
import sys
import threading

from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.text import Text

from ..agent.orchestrator import Orchestrator
from ..agent.types import StepType
from .cancel import CancelController
from .utils import console, make_tag


def tail_text(text: str, max_lines: int) -> Text:
    """取 text 尾部 max_lines 行，确保 Live 区域不超出终端高度。"""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return Text(text)
    tail = "\n".join(lines[-max_lines:])
    return Text("…\n" + tail, style="dim")


class TypewriterDisplay:
    """打字机效果 + 完成后覆写为 Markdown。

    TTY 模式：用 rich Live(transient=True) 追踪流式纯文本输出，
    完成后 stop() 自动清除全部输出，再一次性渲染 Markdown。
    非终端（管道/重定向）直接输出文本。
    """

    def __init__(self):
        self._console = console
        self._is_tty: bool = console.is_terminal
        self._status: Status | None = None
        self._buffer = ""
        self._lock = threading.Lock()
        self._live: Live | None = None

    def start_thinking(self):
        if not self._is_tty:
            return
        with self._lock:
            if self._status is None:
                self._status = Status(
                    Spinner("dots", text=Text(" Thinking...", style="dim")),
                    console=self._console,
                )
                self._status.start()

    def stop_spinner(self):
        if not self._is_tty:
            return
        with self._lock:
            if self._status:
                self._status.stop()
                self._status = None

    def append_token(self, token: str):
        if self._is_tty:
            self.stop_spinner()
        with self._lock:
            self._buffer += token
        if self._is_tty:
            _, height = shutil.get_terminal_size((80, 24))
            max_lines = max(3, height - 2)
            if self._live is None:
                self._live = Live(
                    tail_text(self._buffer, max_lines),
                    console=self._console,
                    refresh_per_second=30,
                    transient=True,
                )
                self._live.start()
            else:
                self._live.update(tail_text(self._buffer, max_lines))
        else:
            sys.stdout.write(token)
            sys.stdout.flush()

    def finish(self) -> str:
        with self._lock:
            result = self._buffer
            self._buffer = ""
            if self._live:
                self._live.stop()
                self._live = None
        if result and self._is_tty:
            console.print(Markdown(result))
        return result

    def abort(self) -> None:
        """打断时清理：停止 spinner 和 Live，不渲染最终 Markdown。"""
        with self._lock:
            if self._status:
                self._status.stop()
                self._status = None
            if self._live:
                self._live.stop()
                self._live = None
            self._buffer = ""


def print_guard(event):
    """打印 Guard 步骤（安全检测结果）。"""
    phase = event.metadata.get("phase", "pre")
    risk = event.metadata.get("risk_level", "safe")
    blocked = event.metadata.get("blocked", False)
    findings = event.metadata.get("findings", [])

    if risk == "safe" and not findings:
        return

    label = "拦截" if blocked else ("警告" if risk == "warning" else "提示")
    color = "bold red" if blocked else ("bold yellow" if risk == "warning" else "dim")
    phase_label = "前置检测" if phase == "pre" else "后置检测"

    console.print(
        Text.assemble(
            ("  ⠂Guard  ", color),
            (f"{phase_label} [{label}]", color),
        )
    )
    if event.content:
        console.print(
            Text.assemble(
                ("          ", ""),
                (event.content, color),
            )
        )
    for f in findings:
        console.print(
            Text.assemble(
                ("          ", ""),
                (f"- 类型: {f.get('type', '?')}, 值: {f.get('value', '')}", "dim"),
            )
        )
    console.print()


def print_think(event, display: TypewriterDisplay):
    """打印 Think 步骤（工具调用决定）。"""
    display.stop_spinner()
    details = event.metadata.get("tool_details", [])
    if not details:
        return
    for d in details:
        name = d.get("name", "unknown")
        source = d.get("source", "unknown")
        category = d.get("category", "")
        args = d.get("args", {})
        tag = make_tag(source, category, name)
        console.print(
            Text.assemble(
                ("  ⠂Think  ", "bold blue"),
                (name, "bold"),
                (f"  {tag}", "dim"),
            )
        )
        for k, v in args.items():
            console.print(
                Text.assemble(
                    ("          ", ""),
                    (f"{k} = {v!r}", "dim"),
                )
            )
    console.print()


def print_act(event, display: TypewriterDisplay):
    """打印 Act 步骤（工具执行结果）。"""
    display.stop_spinner()
    name = event.metadata.get("tool_name", "unknown")
    source = event.metadata.get("tool_source", "unknown")
    category = event.metadata.get("tool_category", "")
    tag = make_tag(source, category, name)
    console.print(
        Text.assemble(
            ("  ⠂Act    ", "bold yellow"),
            (name, "bold"),
            (f"  {tag}", "dim"),
        )
    )
    console.print(
        Text.assemble(
            ("          ", ""),
            (event.content, "yellow"),
        )
    )
    console.print()


def print_config(agent: Orchestrator) -> None:
    """打印当前 Agent 配置信息。"""
    cfg = agent.get_config_info()

    console.print(Rule(style="dim"))
    console.print("[bold]模型配置[/bold]")
    console.print(f"  模型: {cfg['model']}")
    console.print(f"  接口: {cfg['base_url']}")
    mode = cfg.get("approval_mode", "sensitive")
    approval_status = "全部审批" if mode == "full" else "敏感审批"
    console.print(f"  审批: {approval_status}")
    console.print()

    active_skill = cfg.get("skill")
    if active_skill:
        console.print("[bold]当前技能[/bold]")
        console.print(f"  名称: {active_skill['name']}")
        console.print(f"  描述: {active_skill['description']}")
        if active_skill.get("tool_allowlist"):
            console.print(f"  可用工具: {', '.join(active_skill['tool_allowlist'])}")
        console.print()

    console.print("[bold]系统提示词[/bold]")
    console.print(f"  {cfg['system_prompt_preview']}")
    console.print()

    console.print("[bold]工具列表[/bold]")
    for source, names in sorted(cfg["tools_by_source"].items()):
        console.print(Text(f"  [{source}] ({len(names)} 个)"), markup=False)
        for name in sorted(names):
            info = agent._tool_lookup.get(name)
            tag = (
                make_tag(source, info.category if info else "", name) if info else name
            )
            console.print(Text(f"    - {name}  {tag}"), markup=False)
    console.print(f"\n  共 {cfg['total_tools']} 个工具")
    console.print()

    if cfg["mcp_tools"]:
        console.print("[bold]MCP 工具详情[/bold]")
        for mt in cfg["mcp_tools"]:
            console.print(Text(f"    - {mt['name']}  ({mt['category']})"), markup=False)
        console.print()

    if cfg.get("long_term_memory"):
        console.print("[bold]长期记忆[/bold]")
        console.print(f"  历史会话: {cfg['episodic_count']} 条")
        console.print(f"  知识事实: {cfg['semantic_count']} 条")
        console.print()

    console.print(Rule(style="dim"))


def handle_events(agent: Orchestrator, user_input: str):
    """处理 agent 事件流，带 spinner 和打字机效果。

    支持按 ESC 打断：监听期间按 ESC 会停止生成并清理显示。
    Ctrl+C 由 CancelController 转为主线程 KeyboardInterrupt，用于退出程序。
    """
    display = TypewriterDisplay()
    controller = CancelController()
    controller.start()
    resume_esc = False
    finished = False
    try:
        for event in agent.run_stream(user_input):
            if controller.is_cancelled():
                display.abort()
                console.print(
                    Text.assemble(
                        ("  ⠂Interrupted  ", "bold red"),
                        ("已打断当前回复（按 ESC）", "dim"),
                    )
                )
                break
            if resume_esc:
                controller.start()
                resume_esc = False

            if event.step == StepType.GUARD:
                print_guard(event)
            elif event.step == StepType.THINKING:
                display.start_thinking()

            elif event.step == StepType.THINK:
                print_think(event, display)

            elif event.step == StepType.APPROVE:
                # 审批需要用户在终端输入 y/n。先停止 ESC 监听并恢复
                # 终端规范模式（cbreak 下 readline 无法退格/编辑），
                # 审批完成后由后续事件重新启动监听。
                display.stop_spinner()
                controller.stop()
                resume_esc = True

            elif event.step == StepType.ACT:
                print_act(event, display)

            elif event.step == StepType.TOKEN:
                display.append_token(event.content)

            elif event.step == StepType.RESPOND:
                result = display.finish()
                finished = True
                if not result and event.content:
                    console.print(Markdown(event.content))
                    console.print()
    finally:
        controller.stop()
        if not finished:
            display.abort()
