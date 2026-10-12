"""L0 CLI - 终端展示。"""

import queue
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
            sys.stdout.write("思考中...\n")
            sys.stdout.flush()
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
    """打印 Think 步骤（工具调用决定/计划/评估指令）。"""
    display.stop_spinner()
    plan = event.metadata.get("plan")
    if plan:
        console.print(Text("  ⠂Plan    ", style="bold magenta"))
        for line in plan.splitlines():
            console.print(Text(f"          {line}", style="dim"))
        console.print()
        return
    if event.metadata.get("phase") == "verify":
        reason = event.metadata.get("reason", "")
        console.print(Text("  ⠂Verify  ", style="bold cyan"))
        if reason:
            console.print(Text(f"          原因: {reason}", style="dim"))
        if event.content:
            console.print(Text(f"          下一步: {event.content}", style="cyan"))
        console.print()
        return
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
    approval_status = {
        "full": "全部审批",
        "sensitive": "敏感审批",
        "auto": "自动审批（免确认）",
    }.get(mode, mode)
    loop_labels = {
        "manual": "手动 ReAct",
        "plan_execute": "Plan-Execute",
        "langgraph": "LangGraph",
    }
    loop_name = cfg.get("react_loop", "langgraph")
    loop = loop_labels.get(loop_name, loop_name)
    verify_status = (
        f"开启（最多 {cfg.get('verify_max_rounds', 2)} 轮）"
        if cfg.get("verify_enabled")
        else "关闭"
    )
    console.print(f"  审批: {approval_status}")
    console.print(f"  ReAct 循环: {loop}")
    console.print(f"  完成度评估: {verify_status}")
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


class _StreamFeed:
    """把 agent.run_stream 放到后台线程，事件经队列交给主线程消费。

    这样即使 LLM 请求长时间阻塞（首 token 慢/挂起），主线程仍能每 50ms
    轮询 ESC/Ctrl+C，实现立即打断与一次退出的可选关闭。worker 为 daemon，
    进程退出时自动终止，避免残留线程阻塞解释器关闭。
    """

    DONE = "done"
    ERROR = "error"
    EVENT = "event"

    def __init__(self, agent: Orchestrator, user_input: str):
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._agent = agent
        self._user_input = user_input

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for event in self._agent.run_stream(self._user_input):
                if self._stop.is_set():
                    break
                self._q.put((self.EVENT, event))
        except Exception as e:
            self._q.put((self.ERROR, e))
        finally:
            self._q.put((self.DONE, None))

    def next_item(self, timeout: float) -> tuple[str, object] | None:
        """从队列取一个事件；超时返回 None。"""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


def handle_events(agent: Orchestrator, user_input: str):
    """处理 agent 事件流，带 spinner 和打字机效果。

    agent.run_stream 在后台线程运行，事件经队列传给主线程。主循环不断
    轮询队列与键盘，因此 LLM 请求即使阻塞也能：
    - ESC      → 立即打断当前回复，回到 REPL（不退出）。
    - Ctrl+C   → 向主线程抛 KeyboardInterrupt，一次性退出程序。
    """
    display = TypewriterDisplay()
    controller = CancelController()
    controller.start()
    feed = _StreamFeed(agent, user_input)
    feed.start()
    resume_esc = False
    finished = False
    done = False
    interrupted_by_esc = False
    try:
        while True:
            if controller.is_cancelled():
                interrupted_by_esc = True
                break
            if resume_esc:
                controller.start()
                resume_esc = False

            item = feed.next_item(timeout=0.05)
            if item is None:
                continue
            kind, payload = item

            if kind == _StreamFeed.DONE:
                done = True
                break
            if kind == _StreamFeed.ERROR:
                display.abort()
                console.print(
                    f"[bold red][模型调用失败][/bold red] {payload}"[:1200]
                )
                break

            event = payload
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
        feed.stop()
        if not finished:
            display.abort()

    if interrupted_by_esc:
        console.print(
            Text.assemble(
                ("  ⠂Interrupted  ", "bold red"),
                ("已打断当前回复（按 ESC）", "dim"),
            )
        )
