"""L4 记忆 - SQLite 存储层。"""

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = str(Path.home() / ".agent_cli" / "memory.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episodic_created ON episodic_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_created ON semantic_memory(created_at DESC);
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """创建/打开 SQLite 连接，自动初始化 schema。

    Args:
        db_path: 数据库路径，默认 ~/.agent_cli/memory.db。
                 传 ":memory:" 可用内存数据库（测试用）。

    Returns:
        sqlite3.Connection，row_factory 设为 sqlite3.Row。
    """
    path = db_path or os.getenv("AGENT_CLI_DB_PATH", DEFAULT_DB_PATH)
    if path != ":memory:":
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn
