"""L4 SQLite 存储层测试。"""
import sqlite3
from agent_cli.memory.store import get_connection, SCHEMA_SQL, DEFAULT_DB_PATH


class TestGetConnection:
    def test_in_memory_db(self):
        conn = get_connection(":memory:")
        assert isinstance(conn, sqlite3.Connection)

    def test_creates_tables(self):
        conn = get_connection(":memory:")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "episodic_memory" in names
        assert "semantic_memory" in names

    def test_creates_indexes(self):
        conn = get_connection(":memory:")
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        names = [t["name"] for t in indexes]
        assert "idx_episodic_created" in names
        assert "idx_semantic_created" in names

    def test_idempotent_schema(self):
        conn = get_connection(":memory:")
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SCHEMA_SQL)

    def test_row_factory(self):
        conn = get_connection(":memory:")
        assert conn.row_factory == sqlite3.Row

    def test_default_db_path(self):
        assert DEFAULT_DB_PATH.endswith(".agent_cli/memory.db")
