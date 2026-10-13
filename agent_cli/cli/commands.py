"""L0 CLI - chat / run / config 子命令定义。"""

import click

from ..paths import load_env
from .display import handle_events, print_config
from .factory import create_orchestrator
from .repl import run_repl


@click.command()
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm / anthropic / deepseek")
@click.option("--model", "model_name", default=None, help="模型名称，如 spark-x2.5-4b / glm-4-flash")
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
    "--auto-approve", is_flag=True, default=False, help="自动审批：所有工具调用免确认（自主模式）"
)
@click.option(
    "--strategy",
    type=click.Choice(["langgraph", "manual", "plan_execute"]),
    default=None,
    help="Agent 循环策略：langgraph（默认）/ manual / plan_execute",
)
@click.option(
    "--verify", is_flag=True, default=False, help="启用完成度评估：执行后检查目标是否达成，未达成则继续"
)
@click.option(
    "--verify-rounds",
    "verify_max_rounds",
    default=2,
    type=int,
    help="完成度评估最大迭代轮数（默认 2）",
)
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
    model_name,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    auto_approve,
    strategy,
    verify,
    verify_max_rounds,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """启动交互式对话。"""
    load_env()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        model_name=model_name,
        approval=approve,
        auto_approve=auto_approve,
        strategy=strategy,
        enable_verify=verify,
        verify_max_rounds=verify_max_rounds,
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
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm / anthropic / deepseek")
@click.option("--model", "model_name", default=None, help="模型名称，如 spark-x2.5-4b / glm-4-flash")
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
    "--auto-approve", is_flag=True, default=False, help="自动审批：所有工具调用免确认（自主模式）"
)
@click.option(
    "--strategy",
    type=click.Choice(["langgraph", "manual", "plan_execute"]),
    default=None,
    help="Agent 循环策略：langgraph（默认）/ manual / plan_execute",
)
@click.option(
    "--verify", is_flag=True, default=False, help="启用完成度评估：执行后检查目标是否达成，未达成则继续"
)
@click.option(
    "--verify-rounds",
    "verify_max_rounds",
    default=2,
    type=int,
    help="完成度评估最大迭代轮数（默认 2）",
)
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
    model_name,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    auto_approve,
    strategy,
    verify,
    verify_max_rounds,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """单次执行一个问题。"""
    load_env()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        model_name=model_name,
        approval=approve,
        auto_approve=auto_approve,
        strategy=strategy,
        enable_verify=verify,
        verify_max_rounds=verify_max_rounds,
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
@click.option("--provider", default=None, help="模型供应商：openai_compat / glm / anthropic / deepseek")
@click.option("--model", "model_name", default=None, help="模型名称，如 spark-x2.5-4b / glm-4-flash")
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
    "--auto-approve", is_flag=True, default=False, help="自动审批：所有工具调用免确认（自主模式）"
)
@click.option(
    "--strategy",
    type=click.Choice(["langgraph", "manual", "plan_execute"]),
    default=None,
    help="Agent 循环策略：langgraph（默认）/ manual / plan_execute",
)
@click.option(
    "--verify", is_flag=True, default=False, help="启用完成度评估：执行后检查目标是否达成，未达成则继续"
)
@click.option(
    "--verify-rounds",
    "verify_max_rounds",
    default=2,
    type=int,
    help="完成度评估最大迭代轮数（默认 2）",
)
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
    model_name,
    system_prompt,
    mcp_servers,
    mcp_config,
    plugins_dir,
    skill_name,
    approve,
    auto_approve,
    strategy,
    verify,
    verify_max_rounds,
    max_messages,
    max_tokens,
    no_compress,
    no_memory,
):
    """查看当前 Agent 配置信息（不启动对话）。"""
    load_env()
    agent = create_orchestrator(
        system_prompt,
        mcp_servers,
        mcp_config,
        plugins_dir,
        skill_name,
        provider=provider,
        model_name=model_name,
        approval=approve,
        auto_approve=auto_approve,
        strategy=strategy,
        enable_verify=verify,
        verify_max_rounds=verify_max_rounds,
        max_messages=max_messages,
        max_tokens=max_tokens,
        no_compress=no_compress,
        no_memory=no_memory,
    )
    try:
        print_config(agent)
    finally:
        agent.cleanup()
