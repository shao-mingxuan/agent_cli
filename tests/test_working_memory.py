"""L4 WorkingMemory 测试。"""
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent_cli.memory.working import WorkingMemory


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
