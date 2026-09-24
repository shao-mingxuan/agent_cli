"""L0 CLI - chat / run / config 子命令定义。"""

import click
from dotenv import load_dotenv

from .display import handle_events, print_config
from .factory import create_orchestrator
from .repl import run_repl


@click.command()
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm")
@click.option("--system-prompt", "-s", default=None, help="自定义系统提示词")
@click.option(
    "--mcp-server",
    "mcp_servers",
    multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url",
)
@click.option("--mcp-config", default=None, help="MCP 配置文件路径（JSON 格式）")
@click.option("--plugins-dir", default="./plugins", help="插件目录路径")
@click.option("--skill", "skill_name", default=None, help="激活指定技能")
@click.option("--approve", is_flag=True, default=False, help="启用工具调用审批")
@click.option(
    "--max-messages",
    default=None,
    type=int,
    help="上下文滑动窗口大小（保留的最大消息条数，默认 50）",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="上下文 token 预算（超出触发摘要压缩，默认 4000）",
)
@click.option(
    "--no-compress", is_flag=True, default=False, help="禁用摘要压缩，仅使用滑动窗口"
)
@click.option(
    "--no-memory",
    is_flag=True,
    default=False,
    help="禁用长期记忆（不持久化会话摘要和事实）",
)
def chat(
    provider,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """启动交互式对话。"""
    load_dotenv()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        approval=approve,
        max_messages=max_messages,
        max_tokens=max_tokens,
        no_compress=no_compress,
        no_memory=no_memory,
    )
    try:
        run_repl(agent)
    finally:
        agent.cleanup()


@click.command()
@click.option("--prompt", "-p", required=True, help="要发送的问题")
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm")
@click.option("--system-prompt", "-s", default=None, help="自定义系统提示词")
@click.option(
    "--mcp-server",
    "mcp_servers",
    multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url",
)
@click.option("--mcp-config", default=None, help="MCP 配置文件路径（JSON 格式）")
@click.option("--plugins-dir", default="./plugins", help="插件目录路径")
@click.option("--skill", "skill_name", default=None, help="激活指定技能")
@click.option("--approve", is_flag=True, default=False, help="启用工具调用审批")
@click.option(
    "--max-messages",
    default=None,
    type=int,
    help="上下文滑动窗口大小（保留的最大消息条数，默认 50）",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="上下文 token 预算（超出触发摘要压缩，默认 4000）",
)
@click.option(
    "--no-compress", is_flag=True, default=False, help="禁用摘要压缩，仅使用滑动窗口"
)
@click.option(
    "--no-memory",
    is_flag=True,
    default=False,
    help="禁用长期记忆（不持久化会话摘要和事实）",
)
def run(
    prompt,
    provider,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """单次执行一个问题。"""
    load_dotenv()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        approval=approve,
        max_messages=max_messages,
        max_tokens=max_tokens,
        no_compress=no_compress,
        no_memory=no_memory,
    )
    try:
        handle_events(agent, prompt)
    finally:
        agent.cleanup()


@click.command()
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm")
@click.option("--system-prompt", "-s", default=None, help="自定义系统提示词")
@click.option(
    "--mcp-server",
    "mcp_servers",
    multiple=True,
    help="MCP server，格式：stdio:name:command:arg1,arg2 或 sse:name:url",
)
@click.option("--mcp-config", default=None, help="MCP 配置文件路径（JSON 格式）")
@click.option("--plugins-dir", default="./plugins", help="插件目录路径")
@click.option("--skill", "skill_name", default=None, help="激活指定技能")
@click.option("--approve", is_flag=True, default=False, help="启用工具调用审批")
@click.option(
    "--max-messages",
    default=None,
    type=int,
    help="上下文滑动窗口大小（保留的最大消息条数，默认 50）",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="上下文 token 预算（超出触发摘要压缩，默认 4000）",
)
@click.option(
    "--no-compress", is_flag=True, default=False, help="禁用摘要压缩，仅使用滑动窗口"
)
@click.option(
    "--no-memory",
    is_flag=True,
    default=False,
    help="禁用长期记忆（不持久化会话摘要和事实）",
)
def config(
    provider,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """查看当前 Agent 配置信息（不启动对话）。"""
    load_dotenv()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        approval=approve,
        max_messages=max_messages,
        max_tokens=max_tokens,
        no_compress=no_compress,
        no_memory=no_memory,
    )
    try:
        print_config(agent)
    finally:
        agent.cleanup()
