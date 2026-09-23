"""L1 编排 - 长期记忆 Helper 函数。"""

from typing import Any

from langchain_core.messages import SystemMessage


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
    """首次输入时检索语义记忆并注入到 WorkingMemory。"""
    if orch._semantic_memory is None:
        return
    try:
        text = orch._semantic_memory.format_recall(user_input)
        if text:
            orch.memory.add_message(SystemMessage(content=text))
    except Exception:
        pass


def save_session(orch) -> None:
    """提取会话摘要和事实，持久化到 SQLite。"""
    if orch._episodic_memory is None:
        return

    messages = orch.memory.get_messages()
    if len(messages) < 4:
        return

    text = orch.memory._serialize_messages(messages)

    if orch.memory._summarizer:
        try:
            summary = orch.memory._summarizer(text)
            if summary:
                orch._episodic_memory.save_session(orch._session_id, summary)
        except Exception:
            pass

    if orch._fact_extractor:
        try:
            facts = orch._fact_extractor(text)
            for fact in facts:
                emb = None
                if orch._embedder:
                    try:
                        emb = orch._embedder(fact)
                    except Exception:
                        pass
                orch._semantic_memory.store_fact(
                    fact, source=orch._session_id, embedding=emb
                )
        except Exception:
            pass
