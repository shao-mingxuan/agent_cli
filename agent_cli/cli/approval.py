"""L0 CLI - 审批回调。"""

import click
from rich.text import Text

from .utils import console, make_tag


def full_approval_callback(tool_calls: list[dict]) -> list[bool]:
    """严格审批回调：所有工具都需用户确认。"""
    results = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        source = tc.get("source", "unknown")
        category = tc.get("category", "")
        args = tc.get("args", {})
        sensitive = tc.get("sensitive", False)
        tag = make_tag(source, category, name)
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
                    (f"{k} = {v!r}", "dim"),
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


def sensitive_approval_callback(tool_calls: list[dict]) -> list[bool]:
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
        tag = make_tag(source, category, name)
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
                    (f"{k} = {v!r}", "dim"),
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
