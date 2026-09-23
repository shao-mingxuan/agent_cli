"""L4 记忆 - 向量记忆（SQLite + Embedding 检索）。"""

import json
import math
import sqlite3
from typing import Callable

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3


class SemanticMemory:
    """向量记忆：将提取的事实与 embedding 存入 SQLite，按余弦相似度检索。

    支持两种检索模式：
    - 语义检索：有 embedder 时，计算 query embedding 与所有事实的余弦相似度
    - 关键词检索（降级）：无 embedder 或 embedding API 失败时，按关键词匹配
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Callable[[str], list[float]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self._conn = conn
        self._embedder = embedder
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._embedder_failed = False

    def store_fact(
        self,
        content: str,
        source: str = "",
        embedding: list[float] | None = None,
    ) -> None:
        """存储一条事实，自动去重。"""
        existing = self._conn.execute(
            "SELECT id FROM semantic_memory WHERE content = ?", (content,)
        ).fetchone()
        if existing:
            return

        emb_json = json.dumps(embedding) if embedding else None
        self._conn.execute(
            "INSERT INTO semantic_memory (content, source, embedding) VALUES (?, ?, ?)",
            (content, source, emb_json),
        )
        self._conn.commit()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """检索与 query 最相关的事实。"""
        if self._embedder is None or self._embedder_failed:
            return self._keyword_retrieve(query, top_k)

        try:
            query_emb = self._embedder(query)
        except Exception:
            self._embedder_failed = True
            return self._keyword_retrieve(query, top_k)

        rows = self._conn.execute(
            "SELECT content, source, embedding FROM semantic_memory "
            "WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            score = _cosine_similarity(query_emb, emb)
            if score >= self._similarity_threshold:
                scored.append(
                    {
                        "content": row["content"],
                        "source": row["source"],
                        "score": score,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        k = top_k or self._top_k
        return scored[:k]

    def _keyword_retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """降级方案：关键词匹配。"""
        keywords = set(query.lower().split())
        if not keywords:
            return []

        rows = self._conn.execute(
            "SELECT content, source FROM semantic_memory"
        ).fetchall()

        scored = []
        for row in rows:
            content_lower = row["content"].lower()
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                scored.append(
                    {
                        "content": row["content"],
                        "source": row["source"],
                        "score": float(matches),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        k = top_k or self._top_k
        return scored[:k]

    def format_recall(self, query: str, top_k: int | None = None) -> str:
        """检索并格式化为可注入的上下文文本。"""
        facts = self.retrieve(query, top_k)
        if not facts:
            return ""
        lines = [f"- {f['content']}" for f in facts]
        return "[相关知识记忆]\n" + "\n".join(lines)

    def count(self) -> int:
        """返回已存储的事实总数。"""
        row = self._conn.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()
        return row[0] if row else 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
