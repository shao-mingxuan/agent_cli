"""L4 记忆 - 历史会话（SQLite 持久化）。"""

import sqlite3

DEFAULT_RECALL_COUNT = 5
DEFAULT_MAX_ENTRIES = 100


class EpisodicMemory:
    """历史会话记忆：将每次会话摘要持久化到 SQLite，新会话启动时回溯最近的摘要。

    数据流：
    - save_session: 存入 (session_id, summary)
    - recall_recent: 按时间倒序取最近 N 条摘要
    - format_recall: 格式化为可注入 WorkingMemory 的上下文文本
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        recall_count: int = DEFAULT_RECALL_COUNT,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._conn = conn
        self._recall_count = recall_count
        self._max_entries = max_entries

    def save_session(self, session_id: str, summary: str) -> None:
        """存储会话摘要，并清理过旧记录。"""
        self._conn.execute(
            "INSERT INTO episodic_memory (session_id, summary) VALUES (?, ?)",
            (session_id, summary),
        )
        self._conn.execute(
            "DELETE FROM episodic_memory WHERE id NOT IN "
            "(SELECT id FROM episodic_memory ORDER BY created_at DESC LIMIT ?)",
            (self._max_entries,),
        )
        self._conn.commit()

    def recall_recent(self, count: int | None = None) -> list[dict]:
        """返回最近 N 条会话摘要。"""
        n = count or self._recall_count
        rows = self._conn.execute(
            "SELECT session_id, summary, created_at "
            "FROM episodic_memory ORDER BY created_at DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "summary": r["summary"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def format_recall(self, count: int | None = None) -> str:
        """将最近的会话摘要格式化为可注入的上下文文本。"""
        sessions = self.recall_recent(count)
        if not sessions:
            return ""
        lines = []
        for s in sessions:
            lines.append(f"- [{s['created_at']}] {s['summary']}")
        return "[历史会话记忆]\n" + "\n".join(lines)

    def count(self) -> int:
        """返回已存储的会话总数。"""
        row = self._conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()
        return row[0] if row else 0
