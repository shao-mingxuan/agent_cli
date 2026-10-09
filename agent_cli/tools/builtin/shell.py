"""L3 工具 - 执行 shell 命令（类似 myflicker 的命令执行能力）。"""

import subprocess

from langchain.tools import tool

_DEFAULT_TIMEOUT = 60
_MAX_OUTPUT_CHARS = 20000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n... (输出过长，已截断，共 {len(text)} chars)"


@tool
def run_command(command: str, timeout: int = _DEFAULT_TIMEOUT, cwd: str = "") -> str:
    """执行一条本地 shell 命令（通过系统 shell 运行）。可执行任意本地命令，工作目录不受 MCP 限制。

    用于运行测试、构建、git 操作、查看状态等任务。以当前用户权限执行，
    支持管道、重定向、&& 等标准 shell 语法（如 "pytest && git status"）。

    Args:
        command: 要执行的命令行，如 "pytest tests" 或 "git status"
        timeout: 超时秒数，默认 60，超过则中断命令
        cwd: 命令执行的工作目录，留空使用当前工作目录
    """
    if not command.strip():
        return "Error: empty command"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            f"命令执行超过 {timeout}s 已中断。\n"
            f"提示：可适当调大 timeout 参数，或先观察是否有进程卡住。"
        )
    except OSError as e:
        return f"Error: failed to run command: {e}"

    exit_code = proc.returncode
    stdout = _truncate(proc.stdout.rstrip() or "(无输出)")
    stderr = _truncate(proc.stderr.rstrip() or "(无)")
    return f"Exit Code: {exit_code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
