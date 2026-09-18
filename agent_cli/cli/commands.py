"""L0 CLI - chat / run 子命令实现。"""
import sys
import threading

import click
from dotenv import load_dotenv

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.text import Text

from ..agent.orchestrator import Orchestrator
from ..agent.types import StepType
from ..memory.working import WorkingMemory
from ..providers.openai_compat import OpenAICompatProvider
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT


console = Console()


def _make_tag(source: str, category: str, name: str) -> str:
    if category:
        return f"{source}/{category}/{name}"
    return f"{source}/{name}"


class TypewriterDisplay:
    """管理 spinner + 打字机效果的终端显示。

    仅终端（is_terminal=True）时启用 spinner 动画；
    非终端（管道/重定向）直接输出文本，避免 Live 全屏渲染导致卡顿。
    """

    def __init__(self):
        self._console: Console = console
        self._is_tty: bool = console.is_terminal
        self._status: Status | None = None
        self._buffer = ""
        self._live = None
        self._lock = threading.Lock()

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
            if self._is_tty:
                if self._live is None:
                    self._live = _LiveMarkdown(self._console)
                self._live.append(token)
            else:
                sys.stdout.write(token)
                sys.stdout.flush()
            self._buffer += token

    def finish(self) -> str:
        with self._lock:
            if self._live:
                self._live.finish()
                self._live = None
            result = self._buffer
            self._buffer = ""
        return result


class _LiveMarkdown:
    """tty 模式：用 rich Live 渲染逐步增长的 Markdown，兼顾流式和格式化。"""

    def __init__(self, console: Console):
        self._console = console
        self._live: Live | None = None
        self._buffer = ""

    def append(self, token: str):
        self._buffer += token
        if self._live is None:
            self._live = Live(
                Markdown(self._buffer),
                console=self._console,
                refresh_per_second=10,
            )
            self._live.start()
        else:
            self._live.update(Markdown(self._buffer))

    def finish(self):
        if self._live:
            self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
        sys.stdout.write("\n")
        sys.stdout.flush()


def _print_guard(event):
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


def _print_think(event, display: TypewriterDisplay):
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
        tag = _make_tag(source, category, name)
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
                    (f"{k} = {repr(v)}", "dim"),
                )
            )
    console.print()


def _print_act(event, display: TypewriterDisplay):
    """打印 Act 步骤（工具执行结果）。"""
    display.stop_spinner()
    name = event.metadata.get("tool_name", "unknown")
    source = event.metadata.get("tool_source", "unknown")
    category = event.metadata.get("tool_category", "")
    tag = _make_tag(source, category, name)
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


def _handle_events(agent: Orchestrator, user_input: str):
    """处理 agent 事件流，带 spinner 和打字机效果。"""
    display = TypewriterDisplay()

    for event in agent.run_stream(user_input):
        if event.step == StepType.GUARD:
            _print_guard(event)
        elif event.step == StepType.THINKING:
            display.start_thinking()

        elif event.step == StepType.THINK:
            _print_think(event, display)

        elif event.step == StepType.ACT:
            _print_act(event, display)

        elif event.step == StepType.TOKEN:
            display.append_token(event.content)

        elif event.step == StepType.RESPOND:
            result = display.finish()
            if not result and event.content:
                console.print(Markdown(event.content))
                console.print()


def create_orchestrator(system_prompt: str | None = None) -> Orchestrator:
    """创建 Orchestrator 实例。"""
    provider = OpenAICompatProvider()
    memory = WorkingMemory()
    return Orchestrator(
        provider=provider,
        memory=memory,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


def run_repl(agent: Orchestrator) -> None:
    """启动 REPL 交互循环。"""
    console.print("Agent CLI 已启动，输入 /help 查看命令，/exit 退出。")
    console.print(Rule(style="dim"))

    while True:
        try:
            user_input = click.prompt("你", type=str, default="", show_default=False)
            if user_input.strip() == "":
                continue
        except (EOFError, click.exceptions.Abort):
            console.print("\n再见！")
            break

        cmd = user_input.strip().lower()
        if cmd in ("/exit", "/quit"):
            console.print("再见！")
            break
        elif cmd == "/help":
            console.print("[dim]  /exit, /quit  - 退出对话\n  /clear        - 清空对话历史\n  /help         - 显示帮助[/dim]")
            continue
        elif cmd == "/clear":
            agent.reset()
            console.print("[dim]对话历史已清空。[/dim]")
            continue

        _handle_events(agent, user_input)
        console.print(Rule(style="dim"))


@click.command()
@click.option(
    "--system-prompt", "-s", default=None, help="自定义系统提示词"
)
def chat(system_prompt):
    """启动交互式对话。"""
    load_dotenv()
    agent = create_orchestrator(system_prompt)
    run_repl(agent)


@click.command()
@click.option("--prompt", "-p", required=True, help="要发送的问题")
@click.option(
    "--system-prompt", "-s", default=None, help="自定义系统提示词"
)
def run(prompt, system_prompt):
    """单次执行一个问题。"""
    load_dotenv()
    agent = create_orchestrator(system_prompt)
    _handle_events(agent, prompt)
