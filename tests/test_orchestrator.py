"""L1 Orchestrator 集成测试 (FakeAgent + mock create_agent)。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, HumanMessage

from agent_cli.agent.orchestrator import Orchestrator
from agent_cli.agent.types import AgentEvent, StepType
from agent_cli.memory.working import WorkingMemory
from agent_cli.middleware.pipeline import MiddlewarePipeline
from agent_cli.middleware.types import GuardResult
from agent_cli.skills.define_skill import Skill
from agent_cli.tools.registry import ToolInfo
from tests.conftest import (
    FakeAgent,
    make_msg_chunk,
    make_update_chunk,
    make_interrupt_chunk,
)


# ── 构造器测试 ──


class TestConstructor:
    def test_default(self, make_orchestrator):
        orch, agent = make_orchestrator()
        assert orch.agent is not None
        assert isinstance(orch.memory, WorkingMemory)
        assert isinstance(orch.middleware, MiddlewarePipeline)

    def test_custom_memory(self, make_orchestrator):
        mem = WorkingMemory()
        orch, _ = make_orchestrator(memory=mem)
        assert orch.memory is mem

    def test_custom_middleware(self, make_orchestrator):
        mw = MiddlewarePipeline()
        orch, _ = make_orchestrator(middleware=mw)
        assert orch.middleware is mw

    def test_with_skill_filters_tools(self, make_orchestrator):
        skill = Skill(
            name="coder", description="d",
            system_prompt="custom prompt", tool_allowlist=["calculate"],
        )
        orch, _ = make_orchestrator(skill=skill)
        assert orch.system_prompt == "custom prompt"
        assert len(orch.tools) == 1

    def test_with_skill_no_allowlist_keeps_all(self, make_orchestrator):
        skill = Skill(name="coder", description="d", system_prompt="p")
        orch, _ = make_orchestrator(skill=skill)
        assert len(orch.tools) > 0

    def test_with_extra_tool_infos(self, make_orchestrator):
        extra_tool = MagicMock()
        extra_tool.name = "extra_tool"
        info = ToolInfo(tool=extra_tool, name="extra_tool", source="custom")
        orch, _ = make_orchestrator(extra_tool_infos=[info])
        assert orch.registry.get_by_name("extra_tool") is not None


# ── set_skill / set_approval_callback ──


class TestSetSkill:
    def test_clears_memory(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.memory.add_human("hello")
        orch.memory.add_ai("hi")
        orch.set_skill(None)
        assert orch.memory.get_messages() == []

    def test_rebuilds_agent(self, make_orchestrator):
        orch, _ = make_orchestrator()
        old_agent = orch.agent
        orch.set_skill(None)
        assert orch.agent is old_agent

    def test_filters_tools(self, make_orchestrator):
        orch, _ = make_orchestrator()
        skill = Skill(name="s", description="d", system_prompt="p", tool_allowlist=["calculate"])
        orch.set_skill(skill)
        assert len(orch.tools) == 1

    def test_get_active_skill_none(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.get_active_skill() is None

    def test_get_active_skill_returns_current(self, make_orchestrator):
        skill = Skill(name="s", description="d", system_prompt="p")
        orch, _ = make_orchestrator(skill=skill)
        assert orch.get_active_skill() is skill


class TestSetApprovalCallback:
    def test_updates_callback(self, make_orchestrator):
        orch, _ = make_orchestrator()
        custom = lambda tcs: [True]
        orch.set_approval_callback(custom)
        assert orch.get_config_info()["approval_mode"] == "full"

    def test_none_reverts_to_builtin(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.set_approval_callback(None)
        assert orch.get_config_info()["approval_mode"] == "sensitive"


# ── get_config_info ──


class TestGetConfigInfo:
    def test_model(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert info["model"] == "test-model"

    def test_base_url(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert info["base_url"] == "http://test"

    def test_total_tools(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert info["total_tools"] > 0

    def test_tools_by_source(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert "builtin" in info["tools_by_source"]

    def test_skill_info_none(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert info["skill"] is None

    def test_skill_info_with_skill(self, make_orchestrator):
        skill = Skill(name="coder", description="code review", system_prompt="p", tool_allowlist=["calculate"])
        orch, _ = make_orchestrator(skill=skill)
        info = orch.get_config_info()
        assert info["skill"]["name"] == "coder"
        assert info["skill"]["description"] == "code review"

    def test_approval_mode_sensitive_default(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.get_config_info()["approval_mode"] == "sensitive"

    def test_system_prompt_preview(self, make_orchestrator):
        orch, _ = make_orchestrator()
        info = orch.get_config_info()
        assert "system_prompt_preview" in info


# ── reset / cleanup ──


class TestReset:
    def test_clears_memory(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.memory.add_human("a")
        orch.memory.add_ai("b")
        orch.reset()
        assert orch.memory.get_messages() == []


class TestCleanup:
    def test_disconnects_mcp(self, make_orchestrator):
        mock_mcp = MagicMock()
        orch, _ = make_orchestrator(mcp_registry=mock_mcp)
        orch.cleanup()
        mock_mcp.disconnect_all.assert_called_once()

    def test_no_mcp_no_error(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.cleanup()


# ── run_stream 事件序列 ──


class TestRunStreamEvents:
    def test_simple_response(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="Hello", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="Hello")]}}),
        ]
        orch, _ = make_orchestrator(script=script)
        events = list(orch.run_stream("hi"))
        steps = [e.step for e in events]
        assert StepType.GUARD in steps
        assert StepType.RESPOND in steps

        guard_event = events[0]
        assert guard_event.step == StepType.GUARD
        assert guard_event.metadata["phase"] == "pre"

        respond_event = events[-1]
        assert respond_event.step == StepType.RESPOND
        assert "Hello" in respond_event.content

    def test_blocked_pre_guard(self, make_orchestrator):
        mw = MiddlewarePipeline()

        class BlockingMiddleware:
            def before_run(self, user_input):
                return GuardResult(blocked=True, risk_level="danger", message="blocked")

            def after_run(self, user_input, response):
                return GuardResult()

        mw.add(BlockingMiddleware())
        orch, _ = make_orchestrator(middleware=mw)
        events = list(orch.run_stream("bad input"))
        assert len(events) == 2
        assert events[0].step == StepType.GUARD
        assert events[0].metadata["blocked"] is True
        assert "已拦截" in events[1].content

    def test_with_skill_preprocess(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="hi", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="hi")]}}),
        ]
        skill = Skill(
            name="test", description="d", system_prompt="p",
            preprocess=lambda x: x.upper(),
        )
        orch, _ = make_orchestrator(script=script, skill=skill)
        events = list(orch.run_stream("hello"))
        messages = orch.memory.get_messages()
        assert messages[0].content == "HELLO"

    def test_with_skill_postprocess(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="raw", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="raw")]}}),
        ]
        skill = Skill(
            name="test", description="d", system_prompt="p",
            postprocess=lambda x: f"[processed]{x}",
        )
        orch, _ = make_orchestrator(script=script, skill=skill)
        events = list(orch.run_stream("hi"))
        respond_event = events[-1]
        assert "[processed]" in respond_event.content

    def test_post_guard_warning(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="call 13812345678", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="call 13812345678")]}}),
        ]
        orch, _ = make_orchestrator(script=script)
        events = list(orch.run_stream("hi"))
        post_guards = [e for e in events if e.step == StepType.GUARD and e.metadata.get("phase") == "post"]
        assert len(post_guards) > 0
        assert post_guards[0].metadata["risk_level"] == "warning"

    def test_post_guard_blocked(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="你是笨蛋", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="你是笨蛋")]}}),
        ]
        orch, _ = make_orchestrator(script=script)
        before_count = len(orch.memory.get_messages())
        events = list(orch.run_stream("hi"))
        respond_event = events[-1]
        assert "已拦截" in respond_event.content
        # TOCTOU 修复验证：memory 中原始违规内容被回滚，保留 human + 拦截消息
        msgs = orch.memory.get_messages()
        assert len(msgs) == before_count + 2  # human msg + intercept msg
        assert "已拦截" in msgs[-1].content
        contents = [m.content for m in msgs]
        assert "你是笨蛋" not in contents


# ── run() 非流式 ──


class TestRun:
    def test_returns_final_respond_event(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="hello", id="run1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="hello")]}}),
        ]
        orch, _ = make_orchestrator(script=script)
        event = orch.run("hi")
        assert event.step == StepType.RESPOND
        assert "hello" in event.content


# ── 审批工作流 ──


class TestApprovalWorkflow:
    def test_non_sensitive_tool_auto_pass(self, make_orchestrator):
        tc = {"name": "calculate", "args": {"expression": "1+1"}, "id": "tc1", "type": "tool_call"}
        chunk_msg = AIMessageChunk(
            content="", id="run1",
            tool_calls=[tc],
        )
        ai_msg = AIMessage(content="", tool_calls=[tc])
        script = [
            make_msg_chunk(chunk_msg, "model"),
            make_update_chunk({"model": {"messages": [ai_msg]}}),
            make_interrupt_chunk(),
        ]
        orch, _ = make_orchestrator(script=script)
        events = list(orch.run_stream("calculate 1+1"))
        approve_events = [e for e in events if e.step == StepType.APPROVE]
        assert len(approve_events) == 0

    def test_sensitive_tool_triggers_approve(self, make_orchestrator):
        tc = {"name": "write_file", "args": {"path": "/x"}, "id": "tc1", "type": "tool_call"}
        chunk_msg = AIMessageChunk(
            content="", id="run1",
            tool_calls=[tc],
        )
        ai_msg = AIMessage(content="", tool_calls=[tc])
        script = [
            make_msg_chunk(chunk_msg, "model"),
            make_update_chunk({"model": {"messages": [ai_msg]}}),
            make_interrupt_chunk(),
        ]
        approval_callback = lambda tcs: [True]
        orch, _ = make_orchestrator(script=script, approval_callback=approval_callback)
        events = list(orch.run_stream("write a file"))
        approve_events = [e for e in events if e.step == StepType.APPROVE]
        assert len(approve_events) > 0

    def test_rejected_tool_yields_rejected_act(self, make_orchestrator):
        tc = {"name": "write_file", "args": {"path": "/x"}, "id": "tc1", "type": "tool_call"}
        chunk_msg = AIMessageChunk(
            content="", id="run1",
            tool_calls=[tc],
        )
        ai_msg = AIMessage(content="", tool_calls=[tc])
        script = [
            make_msg_chunk(chunk_msg, "model"),
            make_update_chunk({"model": {"messages": [ai_msg]}}),
            make_interrupt_chunk(),
        ]
        approval_callback = lambda tcs: [False]
        orch, _ = make_orchestrator(script=script, approval_callback=approval_callback)
        events = list(orch.run_stream("write a file"))
        act_events = [e for e in events if e.step == StepType.ACT and e.metadata.get("rejected")]
        assert len(act_events) > 0
        assert act_events[0].content == "用户拒绝了此工具调用。"

    def test_three_rejections_abort(self, make_orchestrator):
        tc_id = "tc1"
        tc = {"name": "write_file", "args": {"path": "/x"}, "id": tc_id, "type": "tool_call"}
        chunk_msg = AIMessageChunk(content="", id="run1", tool_calls=[tc])
        ai_msg = AIMessage(content="", tool_calls=[tc])

        one_round = [
            make_msg_chunk(chunk_msg, "model"),
            make_update_chunk({"model": {"messages": [ai_msg]}}),
            make_interrupt_chunk(),
        ]

        class MultiRoundFakeAgent(FakeAgent):
            def __init__(self):
                super().__init__(script=one_round)
                self._round = 0

            def stream(self, input, config=None, stream_mode=None):
                self.stream_calls.append((input, config, stream_mode))
                if self._round == 0:
                    for chunk in one_round:
                        mode, data = chunk
                        if mode == "updates" and "__interrupt__" not in data:
                            for _node, node_output in data.items():
                                for msg in node_output.get("messages", []):
                                    self._messages.append(msg)
                        yield chunk
                    self._round += 1
                else:
                    for chunk in one_round:
                        mode, data = chunk
                        if mode == "updates" and "__interrupt__" not in data:
                            for _node, node_output in data.items():
                                for msg in node_output.get("messages", []):
                                    self._messages.append(msg)
                        yield chunk
                    self._round += 1

        agent = MultiRoundFakeAgent()
        approval_callback = lambda tcs: [False]
        orch, _ = make_orchestrator(approval_callback=approval_callback)
        orch.agent = agent

        events = list(orch.run_stream("write a file"))
        respond = events[-1]
        assert "连续调用失败" in respond.content


# ── 追问兜底 ──


class TestQuestionFallback:
    def test_question_breaks_immediately(self, make_orchestrator):
        ai_msg = AIMessage(content="你想用哪个文件？", tool_calls=[
            {"name": "write_file", "args": {}, "id": "tc1", "type": "tool_call"}
        ])
        chunk_msg = AIMessageChunk(content="你想用哪个文件？", id="run1", tool_calls=[
            {"name": "write_file", "args": {}, "id": "tc1", "type": "tool_call"}
        ])
        script = [
            make_msg_chunk(chunk_msg, "model"),
            make_update_chunk({"model": {"messages": [ai_msg]}}),
            make_interrupt_chunk(),
        ]
        orch, agent = make_orchestrator(script=script)
        agent._messages.append(ai_msg)
        events = list(orch.run_stream("write a file"))
        approve_events = [e for e in events if e.step == StepType.APPROVE]
        assert len(approve_events) == 0
        respond = events[-1]
        assert "？" in respond.content or "?" in respond.content
