"""L4 EpisodicMemory 测试。"""
from agent_cli.memory.store import get_connection
from agent_cli.memory.episodic import EpisodicMemory


class TestEpisodicMemory:
    def test_save_and_recall(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        mem.save_session("s1", "讨论了 React 配置")
        sessions = mem.recall_recent()
        assert len(sessions) == 1
        assert sessions[0]["summary"] == "讨论了 React 配置"
        assert sessions[0]["session_id"] == "s1"

    def test_recall_empty(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        assert mem.recall_recent() == []

    def test_recall_order_desc(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        mem.save_session("s1", "first")
        mem.save_session("s2", "second")
        mem.save_session("s3", "third")
        sessions = mem.recall_recent()
        assert len(sessions) == 3
        assert sessions[0]["summary"] == "third"

    def test_recall_limit(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn, recall_count=2)
        for i in range(5):
            mem.save_session(f"s{i}", f"session {i}")
        sessions = mem.recall_recent()
        assert len(sessions) == 2

    def test_format_recall_empty(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        assert mem.format_recall() == ""

    def test_format_recall_with_data(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        mem.save_session("s1", "discussed Python")
        text = mem.format_recall()
        assert "[历史会话记忆]" in text
        assert "discussed Python" in text

    def test_count(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn)
        assert mem.count() == 0
        mem.save_session("s1", "summary")
        assert mem.count() == 1

    def test_max_entries_cleanup(self):
        conn = get_connection(":memory:")
        mem = EpisodicMemory(conn, max_entries=3)
        for i in range(10):
            mem.save_session(f"s{i}", f"summary {i}")
        assert mem.count() == 3
