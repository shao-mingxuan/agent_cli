"""L4 记忆 - 向量记忆（SQLite + Embedding 检索）。"""

import json
import math
import re
import sqlite3
from typing import Callable

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """将文本拆分为可用于匹配的 token 集合。

    英文/数字按单词拆分（转小写），中文连续段按双字 bigram 拆分，
    同时保留整段，使无 embedding 时的关键词检索对中文同样有效。
    """
    text = text.lower()
    tokens: set[str] = set()
    tokens.update(_ASCII_RE.findall(text))
    for run in _CJK_RE.findall(text):
        if len(run) >= 2:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
        tokens.add(run)
    return tokens


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
        """降级方案：关键词匹配（含中文 bigram）。"""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        rows = self._conn.execute(
            "SELECT content, source FROM semantic_memory"
        ).fetchall()

        scored = []
        for row in rows:
            fact_tokens = _tokens(row["content"])
            matches = len(query_tokens & fact_tokens)
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

    def recent_facts(self, limit: int = 5) -> list[dict]:
        """返回最近存储的 limit 条事实（按入库顺序倒序）。"""
        rows = self._conn.execute(
            "SELECT content, source FROM semantic_memory ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"content": r["content"], "source": r["source"]} for r in rows]

    def format_recent(self, limit: int = 5) -> str:
        """格式化为可注入的最近事实文本（查询无命中时的兜底）。"""
        facts = self.recent_facts(limit)
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
