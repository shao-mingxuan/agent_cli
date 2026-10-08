"""L0 CLI - REPL 交互循环。"""

import click
from rich.rule import Rule
from rich.text import Text

from ..agent.orchestrator import Orchestrator
from ..providers.glm import ZhipuProvider
from ..providers.openai_compat import OpenAICompatProvider
from ..skills.loader import load_skill_registry
from .approval import full_approval_callback, sensitive_approval_callback
from .display import handle_events, print_config
from .factory import create_provider
from .utils import console, make_tag


def read_multiline_input() -> str:
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
            line = click.prompt(prompt_text, type=str, default="", show_default=False)
        except EOFError:
            break
        except click.exceptions.Abort:
            raise  # Ctrl+C 直接退出，由 run_repl 统一处理
        if not line:
            break
        lines.append(line)
        if _brackets_balanced("\n".join(lines)):
            break
    return "\n".join(lines)


def run_repl(agent: Orchestrator) -> None:
    """启动 REPL 交互循环。"""
    import sys

    if not sys.stdin.isatty():
        console.print(
            "[bold red]错误：当前终端不是交互式 TTY，无法启动对话。[/bold red]"
        )
        console.print(
            "[dim]提示：在 macOS 上通过 `make` 运行可能会关闭 stdin。"
            "请直接运行命令：[/dim]"
        )
        console.print(
            "[dim]  venv/bin/agent chat --mcp-config ./mcp.json[/dim]"
        )
        console.print(
            "[dim]或在 Makefile 中使用 `script -q /dev/null` 包装命令。[/dim]"
        )
        return

    console.print("Agent CLI 已启动，输入 /help 查看命令，/exit 退出。")
    console.print(Rule(style="dim"))

    while True:
        try:
            user_input = read_multiline_input()
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
                "  /model <name> - 切换当前模型\n"
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
                tag = make_tag(info.source, info.category, info.name)
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
                    if (
                        agent.get_active_skill()
                        and agent.get_active_skill().name == s.name
                    )
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
            print_config(agent)
            continue
        elif cmd == "/clear":
            agent.reset()
            console.print("[dim]对话历史已清空。[/dim]")
            continue
        elif cmd.startswith("/model"):
            name = user_input.strip()[6:].strip()
            if not name:
                current = getattr(agent.provider, "model_name", "unknown")
                console.print(f"[dim]当前模型: {current}。用法: /model <模型名>[/dim]")
                continue
            wants_glm = name.startswith("glm-")
            is_glm = isinstance(agent.provider, ZhipuProvider)
            try:
                if wants_glm != is_glm:
                    new_provider = create_provider(
                        "glm" if wants_glm else "openai_compat"
                    )
                    agent.set_provider(new_provider, model_name=name)
                    current = getattr(agent.provider, "model_name", "unknown")
                    base_url = getattr(agent.provider, "base_url", "")
                    console.print(
                        f"[dim]已切换到接口: {base_url}，模型: {current}[/dim]"
                    )
                else:
                    agent.set_model(name)
                    current = getattr(agent.provider, "model_name", "unknown")
                    console.print(f"[dim]已切换到模型: {current}[/dim]")
            except Exception as e:
                console.print(f"[bold red]切换模型失败: {e}[/bold red]")
                continue
            console.print(Rule(style="dim"))
            continue
        elif cmd.startswith("/approve"):
            flag = user_input.strip()[8:].strip().lower()
            if flag == "on":
                agent.set_approval_callback(full_approval_callback)
                console.print("[dim]审批模式: 全部工具需手动确认[/dim]")
            elif flag == "off":
                agent.set_approval_callback(sensitive_approval_callback)
                console.print("[dim]审批模式: 仅敏感工具需手动确认[/dim]")
            else:
                mode = agent.get_config_info().get("approval_mode", "sensitive")
                status = "全部审批" if mode == "full" else "敏感审批"
                console.print(
                    f"[dim]当前审批状态: {status}。用法: /approve on|off[/dim]"
                )
            console.print(Rule(style="dim"))
            continue

        try:
            handle_events(agent, user_input)
        except KeyboardInterrupt:
            console.print("\n再见！")
            break
        console.print(Rule(style="dim"))
