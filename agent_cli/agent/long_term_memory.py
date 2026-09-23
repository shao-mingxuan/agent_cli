"""L1 编排 - 长期记忆 Helper 函数。"""

import os
import threading
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# 退出时保存会话记忆的等待上限（秒），可通过环境变量调整。
# LLM 网络调用在后台线程执行，超时即放弃等待，避免阻塞退出流程；
# 但事实提取结果会在线程内立即写入 SQLite，不依赖本次超时。
DEFAULT_SAVE_TIMEOUT = float(os.getenv("AGENT_CLI_MEMORY_TIMEOUT", "10"))


def init_long_term_memory(orch, db_path: str | None, extractor_model: Any) -> None:
    """初始化长期记忆：SQLite 连接、episodic/semantic 记忆、事实提取器。"""
    from ..memory.episodic import EpisodicMemory
    from ..memory.extractor import create_fact_extractor
    from ..memory.semantic import SemanticMemory
    from ..memory.store import get_connection

    orch._db_conn = get_connection(db_path)

    get_embeddings = getattr(orch.provider, "get_embeddings", None)
    if get_embeddings:
        try:
            orch._embedder = get_embeddings()
        except Exception:
            orch._embedder = None

    orch._episodic_memory = EpisodicMemory(orch._db_conn)
    orch._semantic_memory = SemanticMemory(orch._db_conn, embedder=orch._embedder)

    try:
        orch._fact_extractor = create_fact_extractor(extractor_model)
    except Exception:
        orch._fact_extractor = None

    inject_episodic_memory(orch)


def inject_episodic_memory(orch) -> None:
    """注入历史会话回忆到 WorkingMemory。"""
    if orch._episodic_memory is None:
        return
    try:
        text = orch._episodic_memory.format_recall()
        if text:
            orch.memory.add_message(SystemMessage(content=text))
    except Exception:
        pass


def inject_semantic_memory(orch, user_input: str) -> None:
    """首次输入时检索语义记忆并注入到 WorkingMemory。

    查询无命中时兜底注入最近的事实，保证跨会话的关键身份信息
    （如名字/职业）即使面对“你好”这类泛化输入也能被带出。
    """
    if orch._semantic_memory is None:
        return
    try:
        text = orch._semantic_memory.format_recall(user_input)
        if not text:
            text = orch._semantic_memory.format_recent(limit=5)
        if text:
            orch.memory.add_message(SystemMessage(content=text))
    except Exception:
        pass


def save_session(orch, timeout: float = DEFAULT_SAVE_TIMEOUT) -> None:
    """提取会话摘要和事实，持久化到 SQLite。

    事实提取优先执行，结果在线程内立即可靠写入 SQLite；
    会话摘要是尽力而为（网络调用），超时或失败则跳过。
    主线程最多等待 timeout 秒，避免退出流程被阻塞。
    """
    if orch._episodic_memory is None or orch._semantic_memory is None:
        return

    messages = orch.memory.get_messages()
    if not any(
        isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip()
        for m in messages
    ):
        return

    text = orch.memory._serialize_messages(messages)
    deadline = time.monotonic() + timeout

    def _compute() -> None:
        if orch._fact_extractor:
            try:
                facts = orch._fact_extractor(text)
            except Exception:
                facts = []
            for fact in facts:
                fact = fact.strip()
                if not fact:
                    continue
                emb = None
                if orch._embedder:
                    try:
                        emb = orch._embedder(fact)
                    except Exception:
                        pass
                try:
                    orch._semantic_memory.store_fact(
                        fact, source=orch._session_id, embedding=emb
                    )
                except Exception:
                    pass

        if time.monotonic() >= deadline:
            return
        if orch.memory._summarizer:
            try:
                summary = orch.memory._summarizer(text)
                if summary:
                    orch._episodic_memory.save_session(orch._session_id, summary)
            except Exception:
                pass

    worker = threading.Thread(target=_compute, daemon=True)
    worker.start()
    worker.join(timeout)
