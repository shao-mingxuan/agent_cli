"""L2 提示词 - 读取/解析 markdown frontmatter。"""

import os
import re
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析文本开头的 YAML 极简 frontmatter。

    Returns:
        (frontmatter 字典, 正文)。无 frontmatter 时返回 ({}, text)。
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        kv = _KEY_VALUE_RE.match(line)
        if kv:
            meta[kv.group(1)] = _strip_quotes(kv.group(2))
    return meta, text[match.end():]


def load_prompt_file(path: str) -> dict[str, Any] | None:
    """读取单个 prompt 文件，解析 frontmatter。

    返回 dict：{name, description, system_prompt, source}。
    文件不存在或 body 为空时返回 None。
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    meta, body = parse_frontmatter(text)
    if not body.strip():
        return None
    return {
        "name": meta.get("name", os.path.splitext(os.path.basename(path))[0]),
        "description": meta.get("description", ""),
        "system_prompt": body.strip(),
        "source": path,
    }


def load_prompt_dir(prompt_dir: str) -> list[dict[str, Any]]:
    """扫描目录下所有 .md/.txt 文件，解析为 prompt dict 列表。"""
    if not os.path.isdir(prompt_dir):
        return []
    prompts: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(prompt_dir)):
        if not filename.endswith((".md", ".txt")) or filename.startswith("_"):
            continue
        prompt = load_prompt_file(os.path.join(prompt_dir, filename))
        if prompt is not None:
            prompts.append(prompt)
    return prompts
