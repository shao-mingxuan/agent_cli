"""L0 CLI - Orchestrator 工厂。"""

import json

from ..agent.orchestrator import Orchestrator
from ..mcp import MCPRegistry, load_mcp_servers_from_config, parse_mcp_server_spec
from ..memory.working import WorkingMemory
from ..prompts.builtin.default import DEFAULT_SYSTEM_PROMPT
from ..providers.openai_compat import OpenAICompatProvider
from ..skills.loader import load_skill_registry
from ..tools.loader import load_plugins
from .approval import full_approval_callback, sensitive_approval_callback
from .utils import console


def create_orchestrator(
    system_prompt: str | None = None,
    mcp_servers: tuple[str, ...] = (),
    mcp_config: str | None = None,
    plugins_dir: str = "./plugins",
    skill_name: str | None = None,
    skills_dir: str = "./skills",
    approval: bool = False,
    max_messages: int | None = None,
    max_tokens: int | None = None,
    no_compress: bool = False,
    no_memory: bool = False,
) -> Orchestrator:
    """创建 Orchestrator 实例，加载内置 + MCP + 插件工具。"""
    provider = OpenAICompatProvider()
    memory = WorkingMemory()

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
            console.print(
                f"[bold red]MCP 配置文件加载失败 ({mcp_config}): {e}[/bold red]"
            )

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
            console.print(
                f"[dim]插件 '{info.name}' 已加载 ({info.source}/{info.category})[/dim]"
            )

    skill_registry = load_skill_registry(skills_dir)
    active_skill = skill_registry.get(skill_name) if skill_name else None
    if active_skill:
        console.print(
            f"[dim]技能 '{active_skill.name}' 已激活: {active_skill.description}[/dim]"
        )

    approval_cb = full_approval_callback if approval else sensitive_approval_callback

    return Orchestrator(
        provider=provider,
        memory=memory,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        mcp_registry=mcp_registry,
        extra_tool_infos=plugin_infos,
        skill=active_skill,
        approval_callback=approval_cb,
        max_messages=max_messages,
        max_tokens=max_tokens,
        enable_compression=not no_compress,
        enable_long_term_memory=not no_memory,
    )
