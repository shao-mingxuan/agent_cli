"""L0 CLI - chat / run 子命令实现。"""
import json
import math
import shutil
import sys
import threading

import click
from dotenv import load_dotenv

from rich.console import Console, Group
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
from ..mcp import MCPRegistry, parse_mcp_server_spec, load_mcp_servers_from_config
from ..tools.loader import load_plugins
from ..skills.loader import load_skill_registry


console = Console()


def _make_tag(source: str, category: str, name: str) -> str:
    if category:
        return f"{source}/{category}/{name}"
    return f"{source}/{name}"


def _tail_text(text: str, max_lines: int) -> Text:
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
        self._console: Console = console
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
                    _tail_text(self._buffer, max_lines),
                    console=self._console,
                    refresh_per_second=30,
                    transient=True,
                )
                self._live.start()
            else:
                self._live.update(_tail_text(self._buffer, max_lines))
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


def _print_config(agent: Orchestrator) -> None:
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
            tag = _make_tag(source, info.category if info else "", name) if info else name
            console.print(Text(f"    - {name}  {tag}"), markup=False)
    console.print(f"\n  共 {cfg['total_tools']} 个工具")
    console.print()

    if cfg["mcp_tools"]:
        console.print("[bold]MCP 工具详情[/bold]")
        for mt in cfg["mcp_tools"]:
            console.print(Text(f"    - {mt['name']}  ({mt['category']})"), markup=False)
        console.print()

    console.print(Rule(style="dim"))


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

        elif event.step == StepType.APPROVE:
            display.stop_spinner()

        elif event.step == StepType.ACT:
            _print_act(event, display)

        elif event.step == StepType.TOKEN:
            display.append_token(event.content)

        elif event.step == StepType.RESPOND:
            result = display.finish()
            if not result and event.content:
                console.print(Markdown(event.content))
                console.print()


def _full_approval_callback(tool_calls: list[dict]) -> list[bool]:
    """严格审批回调：所有工具都需用户确认。"""
    results = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        source = tc.get("source", "unknown")
        category = tc.get("category", "")
        args = tc.get("args", {})
        sensitive = tc.get("sensitive", False)
        tag = _make_tag(source, category, name)
        label = Text.assemble(
            ("  ⠂Approve  ", "bold magenta"),
            (name, "bold"),
            (f"  {tag}", "dim"),
        )
        if sensitive:
            label.append_text(Text("  [敏感]", style="bold yellow"))
        console.print(label)
        for k, v in args.items():
            console.print(
                Text.assemble(
                    ("          ", ""),
                    (f"{k} = {repr(v)}", "dim"),
                )
            )
        try:
            choice = click.prompt(
                "  [y/n]",
                type=str,
                default="y",
                show_default=False,
                prompt_suffix=" ",
            )
            results.append(choice.strip().lower() in ("y", ""))
        except (EOFError, KeyboardInterrupt):
            console.print()
            results.append(False)
    console.print()
    return results


def _sensitive_approval_callback(tool_calls: list[dict]) -> list[bool]:
    """默认审批回调：敏感工具需用户确认，非敏感工具自动批准。"""
    results = []
    has_prompt = False
    for tc in tool_calls:
        sensitive = tc.get("sensitive", False)
        if not sensitive:
            results.append(True)
            continue
        name = tc.get("name", "unknown")
        source = tc.get("source", "unknown")
        category = tc.get("category", "")
        args = tc.get("args", {})
        tag = _make_tag(source, category, name)
        has_prompt = True
        console.print(
            Text.assemble(
                ("  ⠂Approve  ", "bold magenta"),
                (name, "bold"),
                (f"  {tag}", "dim"),
                ("  [敏感]", "bold yellow"),
            )
        )
        for k, v in args.items():
            console.print(
                Text.assemble(
                    ("          ", ""),
                    (f"{k} = {repr(v)}", "dim"),
                )
            )
        try:
            choice = click.prompt(
                "  [y/n]",
                type=str,
                default="y",
                show_default=False,
                prompt_suffix=" ",
            )
            results.append(choice.strip().lower() in ("y", ""))
        except (EOFError, KeyboardInterrupt):
            console.print()
            results.append(False)
    if has_prompt:
        console.print()
    return results


def create_orchestrator(
    system_prompt: str | None = None,
    mcp_servers: tuple[str, ...] = (),
    mcp_config: str | None = None,
    plugins_dir: str = "./plugins",
    skill_name: str | None = None,
    skills_dir: str = "./skills",
    approval: bool = False,
) -> Orchestrator:
    """创建 Orchestrator 实例，加载内置 + MCP + 插件工具。"""
    provider = OpenAICompatProvider()
    memory = WorkingMemory()

    # 收集所有 MCP server 配置
    all_configs: list = []
    for spec in mcp_servers:
        try:
            all_configs.append(parse_mcp_server_spec(spec))
        except ValueError as e:
            console.print(f"[bold red]MCP server spec 解析失败: {e}[/bold red]")

    if mcp_config:
        try:
            all_configs.extend(load_mcp_servers_from_config(mcp_config))
            console.print(f"[dim]已加载 MCP 配置文件: {mcp_config}[/dim]")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            console.print(f"[bold red]MCP 配置文件加载失败 ({mcp_config}): {e}[/bold red]")

    mcp_registry = None
    if all_configs:
        mcp_registry = MCPRegistry()
        for config in all_configs:
            count = mcp_registry.add_server(config)
            if count > 0:
                console.print(
                    f"[dim]MCP server '{config.name}' 已连接，加载 {count} 个工具[/dim]"
                )

    plugin_infos = load_plugins(plugins_dir)
    if plugin_infos:
        for info in plugin_infos:
            console.print(f"[dim]插件 '{info.name}' 已加载 ({info.source}/{info.category})[/dim]")

    # 加载技能
    skill_registry = load_skill_registry(skills_dir)
    active_skill = skill_registry.get(skill_name) if skill_name else None
    if active_skill:
        console.print(
            f"[dim]技能 '{active_skill.name}' 已激活: {active_skill.description}[/dim]"
        )

    approval_cb = (
        _full_approval_callback if approval else _sensitive_approval_callback
    )

    return Orchestrator(
        provider=provider,
        memory=memory,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        mcp_registry=mcp_registry,
        extra_tool_infos=plugin_infos,
        skill=active_skill,
        approval_callback=approval_cb,
    )


def _read_multiline_input() -> str:
    """读取用户输入，支持多行粘贴。

    当已输入内容中 ASCII 括号未闭合时，继续收集下一行；
    遇到空行或括号闭合后提交。避免粘贴多行代码被终端拆成多次独立输入。
    """

    def _brackets_balanced(text: str) -> bool:
        pairs = {")": "(", "]": "[", "}": "{"}
        stack: list[str] = []
        for ch in text:
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack[-1] != pairs[ch]:
                    return True
                stack.pop()
        return len(stack) == 0

    lines: list[str] = []
    while True:
        try:
            prompt_text = "你" if not lines else "… "
            line = click.prompt(
                prompt_text, type=str, default="", show_default=False
            )
        except (EOFError, click.exceptions.Abort):
            break
        if not line:
            break
        lines.append(line)
        if _brackets_balanced("\n".join(lines)):
            break
    return "\n".join(lines)


def run_repl(agent: Orchestrator) -> None:
    """启动 REPL 交互循环。"""
    console.print("Agent CLI 已启动，输入 /help 查看命令，/exit 退出。")
    console.print(Rule(style="dim"))

    while True:
        try:
            user_input = _read_multiline_input()
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
            console.print(
                "[dim]"
                "  /exit, /quit  - 退出对话\n"
                "  /clear        - 清空对话历史\n"
                "  /config       - 查看当前配置信息\n"
                "  /tools        - 列出所有已注册工具\n"
                "  /skills       - 列出所有可用技能\n"
                "  /skill <name> - 切换到指定技能\n"
                "  /approve on|off - 全部审批 / 仅敏感审批\n"
                "  /help         - 显示帮助"
                "[/dim]"
            )
            continue
        elif cmd == "/tools":
            for name, info in sorted(agent._tool_lookup.items()):
                tag = _make_tag(info.source, info.category, info.name)
                desc = (info.tool.description or "").split("\n")[0][:70]
                console.print(
                    Text.assemble(
                        ("  ", ""),
                        (f"{name:35s}", "bold"),
                        (f"  {tag}", "dim"),
                    )
                )
                if desc:
                    console.print(
                        Text.assemble(
                            ("          ", ""),
                            (desc, "dim"),
                        )
                    )
            console.print(f"\n[dim]共 {len(agent._tool_lookup)} 个工具[/dim]")
            console.print(Rule(style="dim"))
            continue
        elif cmd == "/skills":
            skill_registry = load_skill_registry("./skills")
            for s in skill_registry.list_skills():
                active = (
                    " [bold green]●[/bold green]"
                    if (agent.get_active_skill() and agent.get_active_skill().name == s.name)
                    else ""
                )
                console.print(
                    Text.assemble(
                        ("  ", ""),
                        (f"{s.name:20s}", "bold"),
                        (f"{s.description}", "dim"),
                        (active, ""),
                    )
                )
            console.print(Rule(style="dim"))
            continue
        elif cmd.startswith("/skill "):
            name = user_input.strip()[7:].strip()
            if not name:
                console.print("[dim]用法: /skill <技能名>[/dim]")
                continue
            skill_registry = load_skill_registry("./skills")
            skill = skill_registry.get(name)
            if skill is None:
                console.print(f"[bold red]未知技能: {name}[/bold red]")
                available = ", ".join(skill_registry.list_names())
                console.print(f"[dim]可用技能: {available}[/dim]")
                continue
            agent.set_skill(skill)
            console.print(
                f"[dim]已切换到技能: {skill.name} - {skill.description}[/dim]"
            )
            console.print(Rule(style="dim"))
            continue
        elif cmd == "/config":
            _print_config(agent)
            continue
        elif cmd == "/clear":
            agent.reset()
            console.print("[dim]对话历史已清空。[/dim]")
            continue
        elif cmd.startswith("/approve"):
            flag = user_input.strip()[8:].strip().lower()
            if flag == "on":
                agent.set_approval_callback(_full_approval_callback)
                console.print("[dim]审批模式: 全部工具需手动确认[/dim]")
            elif flag == "off":
                agent.set_approval_callback(_sensitive_approval_callback)
                console.print("[dim]审批模式: 仅敏感工具需手动确认[/dim]")
            else:
                mode = agent.get_config_info().get("approval_mode", "sensitive")
                status = "全部审批" if mode == "full" else "敏感审批"
                console.print(f"[dim]当前审批状态: {status}。用法: /approve on|off[/dim]")
            console.print(Rule(style="dim"))
            continue

        _handle_events(agent, user_input)
        console.print(Rule(style="dim"))


@click.command()
@click.option(
    "--system-prompt", "-s", default=None, help="自定义系统提示词"
)
@click.option(
    "--mcp-server", "mcp_servers", multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url"
)
@click.option(
    "--mcp-config", default=None,
    help="MCP 配置文件路径（JSON 格式）"
)
@click.option(
    "--plugins-dir", default="./plugins", help="插件目录路径"
)
@click.option(
    "--skill", "skill_name", default=None, help="激活指定技能"
)
@click.option(
    "--approve", is_flag=True, default=False, help="启用工具调用审批"
)
def chat(system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approve):
    """启动交互式对话。"""
    load_dotenv()
    agent = create_orchestrator(system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approval=approve)
    try:
        run_repl(agent)
    finally:
        agent.cleanup()


@click.command()
@click.option("--prompt", "-p", required=True, help="要发送的问题")
@click.option(
    "--system-prompt", "-s", default=None, help="自定义系统提示词"
)
@click.option(
    "--mcp-server", "mcp_servers", multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url"
)
@click.option(
    "--mcp-config", default=None,
    help="MCP 配置文件路径（JSON 格式）"
)
@click.option(
    "--plugins-dir", default="./plugins", help="插件目录路径"
)
@click.option(
    "--skill", "skill_name", default=None, help="激活指定技能"
)
@click.option(
    "--approve", is_flag=True, default=False, help="启用工具调用审批"
)
def run(prompt, system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approve):
    """单次执行一个问题。"""
    load_dotenv()
    agent = create_orchestrator(system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approval=approve)
    try:
        _handle_events(agent, prompt)
    finally:
        agent.cleanup()


@click.command()
@click.option(
    "--system-prompt", "-s", default=None, help="自定义系统提示词"
)
@click.option(
    "--mcp-server", "mcp_servers", multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url"
)
@click.option(
    "--mcp-config", default=None,
    help="MCP 配置文件路径（JSON 格式）"
)
@click.option(
    "--plugins-dir", default="./plugins", help="插件目录路径"
)
@click.option(
    "--skill", "skill_name", default=None, help="激活指定技能"
)
@click.option(
    "--approve", is_flag=True, default=False, help="启用工具调用审批"
)
def config(system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approve):
    """查看当前 Agent 配置信息（不启动对话）。"""
    load_dotenv()
    agent = create_orchestrator(system_prompt, mcp_servers, mcp_config, plugins_dir, skill_name, approval=approve)
    try:
        _print_config(agent)
    finally:
        agent.cleanup()
