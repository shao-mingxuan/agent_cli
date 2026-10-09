"""L3 工具 - 本地文件读写（类似 myflicker 的文件工具集）。"""

import fnmatch
import os

from langchain.tools import tool

_MAX_READ_BYTES = 2 * 1024 * 1024
_MAX_READ_LINES = 10000
_MAX_LIST_ENTRIES = 300
_MAX_SEARCH_FILES = 20000


def _read_text(path: str) -> str | None:
    size = os.path.getsize(path)
    if size > _MAX_READ_BYTES:
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


@tool
def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """读取文件的指定行区间并返回内容，适合查看代码/文本/配置。可访问任意本地路径（不受 MCP 目录限制）。

    文件以 UTF-8 读取，超大文件只读取单次请求的行区间，避免一次性输出过多。

    Args:
        path: 要读取的文件路径（绝对路径或相对路径）
        offset: 起始行号（从 1 开始），默认 0 表示从文件开头读取
        limit: 最多读取的行数，默认 0 表示读取全部（最多 10000 行）
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: is a directory, not a file: {path}"

    content = _read_text(path)
    if content is None:
        return f"Error: file too large (> {_MAX_READ_BYTES} bytes), read in smaller ranges: {path}"

    lines = content.splitlines()
    total = len(lines)
    if offset < 1:
        offset = 1
    if limit <= 0:
        limit = _MAX_READ_LINES
    limit = max(1, min(limit, _MAX_READ_LINES))

    start = min(offset - 1, total)
    end = min(start + limit, total)
    selected = lines[start:end]

    header = f"({total} lines total)"
    if start == 0 and end == total:
        header = f"{total} lines"
    body = "\n".join(selected)
    truncated = f"\n... 已截断，共 {total} 行 ..." if end < total else ""
    return f"{header}\n{body}{truncated}"


@tool
def write_file(path: str, content: str) -> str:
    """写入（或覆盖）一个文件，必要时自动创建父目录。可写入任意本地路径（不受 MCP 目录限制）。

    用于创建新文件或完整重写已有文件；小范围修改请优先用 edit_file。

    Args:
        path: 目标文件路径（绝对路径或相对路径）
        content: 要写入的完整文件内容
    """
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {os.path.abspath(path)}（{len(content)} bytes）"
    except OSError as e:
        return f"Error: write failed: {e}"


@tool
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """在文件中精确替换字符串，用于小范围修改。可修改任意本地文件（不受 MCP 目录限制）。

    要求 old_string 与文件内容完全一致（含缩进/换行）。默认要求唯一匹配，
    若匹配多次需指定 replace_all=True 或提供更多上下文。

    Args:
        path: 要修改的文件路径
        old_string: 要被替换的原文（必须与文件内容精确匹配）
        new_string: 替换后的新文本
        replace_all: 为 True 时替换所有匹配处；默认为 False 只替换唯一匹配处
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: is a directory, not a file: {path}"

    content = _read_text(path)
    if content is None:
        return f"Error: file too large, cannot edit: {path}"

    occurrences = content.count(old_string)
    if occurrences == 0:
        total_lines = len(content.splitlines())
        return (
            f"Error: old_string not found in file ({total_lines} lines). "
            "请提供与文件内容完全一致（含缩进/换行）的片段。"
        )
    if occurrences > 1 and not replace_all:
        return (
            f"Error: old_string appears {occurrences} times in file. "
            "提供更多上下文使其唯一，或设置 replace_all=True。"
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        return f"Error: write failed: {e}"
    return f"已修改 {path}：替换 {occurrences if replace_all else 1} 处，共 {len(old_string)} → {len(new_string)} 字符"


@tool
def list_directory(path: str = ".") -> str:
    """列出目录下的条目（文件名、类型、大小），不递归。可访问任意本地目录（不受 MCP 目录限制）。

    适合先查看目录结构再决定读取/修改哪个文件。

    Args:
        path: 要列出的目录路径，默认当前目录
    """
    if not os.path.exists(path):
        return f"Error: path not found: {path}"
    if not os.path.isdir(path):
        return f"Error: not a directory: {path}"

    try:
        entries = sorted(os.listdir(path))
    except OSError as e:
        return f"Error: cannot list directory: {e}"

    if len(entries) > _MAX_LIST_ENTRIES:
        entries = entries[:_MAX_LIST_ENTRIES]
        note = f"\n... 条目过多，仅显示前 {_MAX_LIST_ENTRIES} 个（共更多）。"
    else:
        note = ""

    lines = [f"目录 {os.path.abspath(path)}:"]
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"  [目录] {name}/")
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = -1
            lines.append(f"  [文件] {name}  ({size} bytes)")
    return "\n".join(lines) + note


@tool
def search_files(directory: str, pattern: str, max_results: int = 50) -> str:
    """在目录下递归搜索文件名匹配给定 glob 模式的文件。可搜索任意本地目录（不受 MCP 目录限制）。

    只按文件名（含相对子目录路径）匹配，不匹配文件内容。常用模式如
    *.py、test_*.py、*.md、*.txt。

    Args:
        directory: 搜索的起始目录
        pattern: 文件名 glob 模式，如 "*.py"
        max_results: 最多返回的匹配结果数，默认 50
    """
    if not os.path.isdir(directory):
        return f"Error: directory not found: {directory}"

    max_results = max(1, min(max_results, 200))
    matches = []
    scanned = 0
    skipped = False
    for root, dirs, files in os.walk(directory):
        if scanned >= _MAX_SEARCH_FILES:
            skipped = True
            break
        dirs.sort()
        for name in sorted(files):
            scanned += 1
            if fnmatch.fnmatch(name, pattern):
                rel = os.path.relpath(os.path.join(root, name), directory)
                matches.append(rel)
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"在 {directory} 下未找到匹配 {pattern} 的文件"

    lines = [f"找到 {len(matches)} 个匹配 {pattern} 的文件:"]
    lines.extend(f"  - {m}" for m in matches)
    if skipped or len(matches) >= max_results:
        lines.append(f"... 结果过多，仅显示前 {len(matches)} 个。")
    return "\n".join(lines)
