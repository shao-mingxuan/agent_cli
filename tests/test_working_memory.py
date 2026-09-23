"""L4 WorkingMemory 测试。"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage

from agent_cli.memory.working import (
    WorkingMemory,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_COMPRESSION_THRESHOLD,
    DEFAULT_KEEP_RECENT,
)


class TestWorkingMemory:
    def test_empty_memory_get_messages(self):
        assert WorkingMemory().get_messages() == []

    def test_add_human(self):
        mem = WorkingMemory()
        mem.add_human("hi")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "hi"

    def test_add_ai(self):
        mem = WorkingMemory()
        mem.add_ai("bye")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)

    def test_add_message(self):
        mem = WorkingMemory()
        msg = HumanMessage(content="test")
        mem.add_message(msg)
        assert mem.get_messages()[0] is msg

    def test_add_messages_bulk(self):
        mem = WorkingMemory()
        mem.add_messages([HumanMessage(content="a"), AIMessage(content="b")])
        assert len(mem.get_messages()) == 2

    def test_get_messages_returns_copy(self):
        mem = WorkingMemory()
        mem.add_human("x")
        msgs = mem.get_messages()
        msgs.clear()
        assert len(mem.get_messages()) == 1

    def test_clear_empties(self):
        mem = WorkingMemory()
        mem.add_human("a")
        mem.add_ai("b")
        mem.add_human("c")
        mem.clear()
        assert mem.get_messages() == []

    def test_order_preserved(self):
        mem = WorkingMemory()
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.add_human("h2")
        msgs = mem.get_messages()
        assert [m.content for m in msgs] == ["h1", "a1", "h2"]

    def test_multiple_adds_accumulate(self):
        mem = WorkingMemory()
        mem.add_human("a")
        mem.add_ai("b")
        mem.add_human("c")
        assert len(mem.get_messages()) == 3

    def test_add_human_returns_none(self):
        mem = WorkingMemory()
        assert mem.add_human("x") is None

    def test_add_ai_returns_none(self):
        mem = WorkingMemory()
        assert mem.add_ai("x") is None

    def test_clear_returns_none(self):
        mem = WorkingMemory()
        assert mem.clear() is None

    def test_rollback_to_discards_newer_messages(self):
        mem = WorkingMemory()
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.add_human("h2")
        mem.rollback_to(2)
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert [m.content for m in msgs] == ["h1", "a1"]

    def test_rollback_to_noop_when_n_equals_current(self):
        mem = WorkingMemory()
        mem.add_human("h1")
        mem.rollback_to(1)
        assert len(mem.get_messages()) == 1

    def test_rollback_to_noop_when_n_greater_than_current(self):
        mem = WorkingMemory()
        mem.add_human("h1")
        mem.rollback_to(5)
        assert len(mem.get_messages()) == 1

    def test_rollback_to_zero_clears(self):
        mem = WorkingMemory()
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.rollback_to(0)
        assert mem.get_messages() == []


class TestSlidingWindow:
    def test_default_max_messages(self):
        assert DEFAULT_MAX_MESSAGES == 50
        mem = WorkingMemory()
        assert mem.max_messages == DEFAULT_MAX_MESSAGES

    def test_custom_max_messages(self):
        mem = WorkingMemory(max_messages=10)
        assert mem.max_messages == 10

    def test_trims_when_exceeding_max(self):
        mem = WorkingMemory(max_messages=4)
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.add_human("h2")
        mem.add_ai("a2")
        mem.add_human("h3")
        msgs = mem.get_messages()
        assert len(msgs) == 4
        # h1 should be evicted
        contents = [m.content for m in msgs]
        assert "h1" not in contents
        assert "h3" in contents

    def test_system_message_always_retained(self):
        mem = WorkingMemory(max_messages=3)
        mem.add_message(SystemMessage(content="sys"))
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.add_human("h2")
        mem.add_ai("a2")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0].content == "sys"
        assert msgs[1].content == "h2"
        assert msgs[2].content == "a2"

    def test_multiple_system_messages_retained(self):
        mem = WorkingMemory(max_messages=4)
        mem.add_message(SystemMessage(content="sys1"))
        mem.add_message(SystemMessage(content="sys2"))
        mem.add_human("h1")
        mem.add_ai("a1")
        mem.add_human("h2")
        mem.add_ai("a2")
        msgs = mem.get_messages()
        contents = [m.content for m in msgs]
        assert "sys1" in contents
        assert "sys2" in contents
        assert "h1" not in contents  # evicted
        assert "a2" in contents  # recent

    def test_no_trim_when_under_limit(self):
        mem = WorkingMemory(max_messages=10)
        mem.add_human("h1")
        mem.add_ai("a1")
        assert len(mem.get_messages()) == 2

    def test_set_max_messages_triggers_trim(self):
        mem = WorkingMemory(max_messages=100)
        for i in range(10):
            mem.add_human(f"h{i}")
        assert len(mem.get_messages()) == 10
        mem.max_messages = 4
        assert len(mem.get_messages()) == 4
        contents = [m.content for m in mem.get_messages()]
        assert "h9" in contents
        assert "h0" not in contents

    def test_add_message_triggers_trim(self):
        mem = WorkingMemory(max_messages=3)
        mem.add_message(HumanMessage(content="h1"))
        mem.add_message(AIMessage(content="a1"))
        mem.add_message(HumanMessage(content="h2"))
        mem.add_message(AIMessage(content="a2"))
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert "h1" not in [m.content for m in msgs]

    def test_add_messages_trims_after_bulk(self):
        mem = WorkingMemory(max_messages=5)
        msgs = [HumanMessage(content=f"h{i}") for i in range(10)]
        mem.add_messages(msgs)
        assert len(mem.get_messages()) == 5
        contents = [m.content for m in mem.get_messages()]
        assert "h9" in contents
        assert "h0" not in contents

    def test_max_messages_zero_evicts_all_non_system(self):
        mem = WorkingMemory(max_messages=1)
        mem.add_message(SystemMessage(content="sys"))
        mem.add_human("h1")
        mem.add_ai("a1")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "sys"

    def test_no_system_messages_simple_fifo(self):
        mem = WorkingMemory(max_messages=3)
        mem.add_human("a")
        mem.add_ai("b")
        mem.add_human("c")
        mem.add_ai("d")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert [m.content for m in msgs] == ["b", "c", "d"]


class TestTokenEstimation:
    def test_estimate_tokens_positive(self):
        mem = WorkingMemory()
        mem.add_human("hello world")
        assert mem._estimate_tokens(mem.get_messages()) > 0

    def test_estimate_tokens_empty(self):
        mem = WorkingMemory()
        assert mem._estimate_tokens([]) == 0

    def test_estimate_tokens_includes_tool_calls(self):
        mem = WorkingMemory()
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"path": "/x"}, "id": "tc1", "type": "tool_call"}],
        )
        tokens = mem._estimate_tokens([msg])
        assert tokens > 0

    def test_estimate_tokens_non_string_content(self):
        mem = WorkingMemory()
        msg = ToolMessage(content={"key": "val"}, tool_call_id="tc1")
        tokens = mem._estimate_tokens([msg])
        assert tokens > 0


class TestSerializeMessages:
    def test_basic_serialization(self):
        mem = WorkingMemory()
        text = mem._serialize_messages([
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ])
        assert "Human: hello" in text
        assert "AI: hi there" in text

    def test_serialization_with_tool_calls(self):
        mem = WorkingMemory()
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"path": "/x"}, "id": "tc1", "type": "tool_call"}],
        )
        text = mem._serialize_messages([msg])
        assert "write_file" in text
        assert "调用" in text

    def test_serialization_non_string_content(self):
        mem = WorkingMemory()
        msg = ToolMessage(content={"result": 42}, tool_call_id="tc1")
        text = mem._serialize_messages([msg])
        assert "Tool" in text
        assert "42" in text


class TestTokenCompression:
    def test_no_summarizer_no_compress(self):
        mem = WorkingMemory(max_tokens=1, compression_threshold=2)
        for i in range(10):
            mem.add_human(f"message {i}")
        msgs = mem.get_messages()
        assert all(not isinstance(m, SystemMessage) or m.content == "" for m in msgs)
        assert len(msgs) == 10

    def test_compress_triggers_on_token_overflow(self):
        calls = []
        def summarizer(text):
            calls.append(text)
            return "summary of conversation"
        mem = WorkingMemory(
            max_tokens=10,
            compression_threshold=3,
            keep_recent=2,
            summarizer=summarizer,
        )
        for i in range(5):
            mem.add_human(f"long message number {i} with lots of tokens")
        msgs = mem.get_messages()
        assert len(calls) >= 1
        summary_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[历史摘要]" in m.content]
        assert len(summary_msgs) == 1
        non_system = [m for m in msgs if not isinstance(m, SystemMessage)]
        assert len(non_system) <= 2

    def test_no_compress_under_threshold(self):
        calls = []
        def summarizer(text):
            calls.append(text)
            return "summary"
        mem = WorkingMemory(
            max_tokens=1,
            compression_threshold=20,
            keep_recent=2,
            summarizer=summarizer,
        )
        mem.add_human("short")
        mem.add_ai("reply")
        assert len(calls) == 0

    def test_no_compress_under_token_limit(self):
        calls = []
        def summarizer(text):
            calls.append(text)
            return "summary"
        mem = WorkingMemory(
            max_tokens=10000,
            compression_threshold=2,
            keep_recent=2,
            summarizer=summarizer,
        )
        for i in range(5):
            mem.add_human(f"msg {i}")
        assert len(calls) == 0

    def test_summarizer_failure_skips_compress(self):
        def bad_summarizer(text):
            raise RuntimeError("LLM unavailable")
        mem = WorkingMemory(
            max_tokens=1,
            compression_threshold=3,
            keep_recent=2,
            summarizer=bad_summarizer,
        )
        for i in range(5):
            mem.add_human(f"message {i}")
        msgs = mem.get_messages()
        summary_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[历史摘要]" in m.content]
        assert len(summary_msgs) == 0

    def test_keep_recent_preserves_latest(self):
        def summarizer(text):
            return "summary"
        mem = WorkingMemory(
            max_tokens=10,
            compression_threshold=3,
            keep_recent=3,
            summarizer=summarizer,
        )
        for i in range(6):
            mem.add_human(f"msg_{i}")
        msgs = mem.get_messages()
        non_system = [m for m in msgs if not isinstance(m, SystemMessage)]
        contents = [m.content for m in non_system]
        assert "msg_5" in contents
        assert "msg_4" in contents
        assert "msg_3" in contents

    def test_system_message_retained_after_compress(self):
        def summarizer(text):
            return "summary"
        mem = WorkingMemory(
            max_tokens=10,
            compression_threshold=3,
            keep_recent=2,
            summarizer=summarizer,
        )
        mem.add_message(SystemMessage(content="sys prompt"))
        for i in range(5):
            mem.add_human(f"msg_{i}")
        msgs = mem.get_messages()
        assert msgs[0].content == "sys prompt"

    def test_rolling_compression(self):
        calls = []
        def summarizer(text):
            calls.append(text)
            return f"summary #{len(calls)}"
        mem = WorkingMemory(
            max_tokens=10,
            compression_threshold=3,
            keep_recent=2,
            summarizer=summarizer,
        )
        for i in range(10):
            mem.add_human(f"long message {i} " * 5)
        msgs = mem.get_messages()
        assert len(calls) >= 2
        summary_msgs = [m for m in msgs if isinstance(m, SystemMessage) and "[历史摘要]" in m.content]
        assert len(summary_msgs) == 1

    def test_too_few_to_compress_skips(self):
        calls = []
        def summarizer(text):
            calls.append(text)
            return "summary"
        mem = WorkingMemory(
            max_tokens=1,
            compression_threshold=3,
            keep_recent=10,
            summarizer=summarizer,
        )
        mem.add_human("a")
        mem.add_ai("b")
        mem.add_human("c")
        assert len(calls) == 0

    def test_compression_enabled_property(self):
        mem = WorkingMemory()
        assert mem.compression_enabled is False
        mem._summarizer = lambda x: "summary"
        assert mem.compression_enabled is True

    def test_max_tokens_property(self):
        mem = WorkingMemory()
        assert mem.max_tokens == DEFAULT_MAX_TOKENS
        mem.max_tokens = 2000
        assert mem.max_tokens == 2000
