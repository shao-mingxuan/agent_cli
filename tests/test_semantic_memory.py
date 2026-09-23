"""L4 SemanticMemory 测试。"""

from agent_cli.memory.semantic import SemanticMemory, _cosine_similarity
from agent_cli.memory.store import get_connection


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-9

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSemanticMemoryStore:
    def test_store_and_retrieve_with_embedding(self):
        conn = get_connection(":memory:")
        embedder = lambda text: [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]
        mem = SemanticMemory(conn, embedder=embedder, similarity_threshold=0.5)
        mem.store_fact("用户喜欢 Python", source="s1", embedding=[1.0, 0.0])
        results = mem.retrieve("python programming")
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_deduplication(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn)
        mem.store_fact("用户喜欢 Python")
        mem.store_fact("用户喜欢 Python")
        assert mem.count() == 1

    def test_store_without_embedding(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn)
        mem.store_fact("some fact", source="s1")
        assert mem.count() == 1


class TestSemanticMemoryRetrieve:
    def test_keyword_fallback_no_embedder(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn, embedder=None)
        mem.store_fact("用户喜欢 Python 编程")
        mem.store_fact("项目使用 React 框架")
        results = mem.retrieve("python")
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_keyword_fallback_on_embedder_failure(self):
        conn = get_connection(":memory:")

        def bad_embedder(text):
            raise RuntimeError("API unavailable")

        mem = SemanticMemory(conn, embedder=bad_embedder)
        mem.store_fact("用户喜欢 Python", embedding=[1.0])
        results = mem.retrieve("python")
        assert len(results) == 1
        assert mem._embedder_failed is True

    def test_embedder_failure_caches(self):
        conn = get_connection(":memory:")
        call_count = [0]

        def bad_embedder(text):
            call_count[0] += 1
            raise RuntimeError("fail")

        mem = SemanticMemory(conn, embedder=bad_embedder)
        mem.retrieve("query1")
        mem.retrieve("query2")
        assert call_count[0] == 1

    def test_empty_query(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn, embedder=None)
        mem.store_fact("some fact")
        assert mem.retrieve("") == []

    def test_format_recall_empty(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn)
        assert mem.format_recall("anything") == ""

    def test_format_recall_with_data(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn, embedder=None)
        mem.store_fact("用户偏好 TypeScript")
        text = mem.format_recall("typescript")
        assert "[相关知识记忆]" in text
        assert "TypeScript" in text

    def test_keyword_fallback_cjk_matches_bigram(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn, embedder=None)
        mem.store_fact("用户的名字是 peter")
        mem.store_fact("用户的职业是前端工程师")
        results = mem.retrieve("我叫 peter 今年 18 岁")
        assert len(results) == 1
        assert "peter" in results[0]["content"]

    def test_format_recent(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn, embedder=None)
        mem.store_fact("用户的名字是 peter")
        mem.store_fact("用户喜欢钓鱼")
        text = mem.format_recent(limit=2)
        assert "peter" in text
        assert "钓鱼" in text
        assert text.startswith("[相关知识记忆]")

    def test_similarity_threshold(self):
        conn = get_connection(":memory:")
        embedder = lambda text: [0.1, 0.9]
        mem = SemanticMemory(conn, embedder=embedder, similarity_threshold=0.99)
        mem.store_fact("fact 1", embedding=[0.9, 0.1])
        results = mem.retrieve("query")
        assert len(results) == 0

    def test_count(self):
        conn = get_connection(":memory:")
        mem = SemanticMemory(conn)
        assert mem.count() == 0
        mem.store_fact("a")
        mem.store_fact("b")
        assert mem.count() == 2
