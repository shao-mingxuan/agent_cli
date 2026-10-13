"""L0 路径 - 全局资源目录解析。

让 agent 命令在任何工作目录可用：.env / skills / plugins / mcp.json
按以下优先级查找（先到先用）：

1. AGENT_HOME 环境变量指定目录
2. 用户配置目录 $XDG_CONFIG_HOME/agent_cli（macOS/Linux 默认 ~/.config/agent_cli）
3. 项目根目录（安装包所在目录）
4. 当前工作目录
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR_NAME = "agent_cli"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    """返回用户配置目录（$XDG_CONFIG_HOME/agent_cli 或 ~/.config/agent_cli）。"""
    base = os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / APP_DIR_NAME


def _search_roots() -> list[Path]:
    roots: list[Path] = []
    if os.getenv("AGENT_HOME"):
        roots.append(Path(os.environ["AGENT_HOME"]))
    roots.append(config_dir())
    roots.append(PROJECT_ROOT)
    return roots


def find_resource(name: str) -> Path | None:
    """在 AGENT_HOME → 用户配置目录 → 项目根 → 当前目录中查找资源。

    name 可以是文件（如 mcp.json）或目录（如 skills）。返回第一个存在的路径，
    全部不存在时返回 None。
    """
    for root in _search_roots():
        candidate = root / name
        if candidate.exists():
            return candidate
    cwd_candidate = Path.cwd() / name
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def resolve_dir(name: str, fallback: str) -> str:
    """返回存在的资源目录路径，找不到时使用 fallback（保持向后兼容）。"""
    found = find_resource(name)
    return str(found) if found else fallback


def plugins_dir() -> str:
    return resolve_dir("plugins", "./plugins")


def skills_dir() -> str:
    return resolve_dir("skills", "./skills")


def mcp_config_path() -> str | None:
    """返回第一个存在的 mcp.json 路径，未找到返回 None。"""
    found = find_resource("mcp.json")
    return str(found) if found else None


def load_env() -> None:
    """按优先级链式加载 .env：AGENT_HOME → 用户配置目录 → 项目根 → 当前目录。

    高优先级先加载，后加载的 .env 不覆盖已存在的环境变量（dotenv override=False）。
    """
    from dotenv import load_dotenv

    for root in [*_search_roots(), Path.cwd()]:
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
