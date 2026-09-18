"""L0 入口 - 仅命令解析。"""
import click

from .cli.commands import chat, run


@click.group()
def cli():
    """Agent CLI - 基于 LangChain 的命令行助手。"""
    pass


cli.add_command(chat)
cli.add_command(run)


if __name__ == "__main__":
    cli()
