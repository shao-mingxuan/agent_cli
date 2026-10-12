"""自主决策增强测试：自动审批 + 完成度评估闭环 (verifier)。"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from agent_cli.agent.types import StepType
from agent_cli.tools.registry import ToolInfo
from tests.conftest import make_msg_chunk, make_update_chunk


def _ready_bound_model(orch, responses):
    bound = MagicMock()
    bound.invoke.side_effect = responses
    orch.model.bind_tools.return_value = bound
    return bound


def _humans(orch):
    return [m.content for m in orch.memory.get_messages() if m.type == "human"]


class TestApprovalMode:
    def test_default_sensitive(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.get_config_info()["approval_mode"] == "sensitive"

    def test_auto_mode_reported(self, make_orchestrator):
        orch, _ = make_orchestrator(approval_mode="auto")
        assert orch.get_config_info()["approval_mode"] == "auto"

    def test_full_mode_reported(self, make_orchestrator):
        orch, _ = make_orchestrator(approval_mode="full")
        assert orch.get_config_info()["approval_mode"] == "full"

    def test_set_approval_callback_with_mode(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.set_approval_callback(lambda tcs: [True], mode="auto")
        assert orch.get_config_info()["approval_mode"] == "auto"
        orch.set_approval_callback(lambda tcs: [True], mode="sensitive")
        assert orch.get_config_info()["approval_mode"] == "sensitive"


class TestAutoApprovalCallback:
    def test_cli_auto_callback_approves_all(self):
        from agent_cli.cli.approval import auto_approval_callback

        calls = [
            {"name": "run_command", "sensitive": True},
            {"name": "read_file", "sensitive": True},
            {"name": "calculate", "sensitive": False},
        ]
        assert auto_approval_callback(calls) == [True, True, True]

    def test_auto_mode_runs_sensitive_tool(self, make_orchestrator):
        tool = MagicMock()
        tool.name = "run_command"
        tool.invoke.return_value = "ok"
        info = ToolInfo(
            tool=tool, name="run_command", source="builtin", category="shell"
        )
        orch, _ = make_orchestrator(
            use_manual_react=True,
            approval_callback=lambda tcs: [True] * len(tcs),
            approval_mode="auto",
            extra_tool_infos=[info],
        )
        tc = {
            "name": "run_command",
            "args": {"cmd": "ls"},
            "id": "tc1",
            "type": "tool_call",
        }
        _ready_bound_model(
            orch,
            [AIMessage(content="", tool_calls=[tc]), AIMessage(content="done")],
        )
        events = list(orch.run_stream("run it"))
        assert any(e.step == StepType.APPROVE for e in events)
        tool.invoke.assert_called_once_with({"cmd": "ls"})


class TestVerifyLoop:
    def _manual(self, make_orchestrator, bound_responses, verdict_responses):
        orch, _ = make_orchestrator(use_manual_react=True, enable_verify=True)
        _ready_bound_model(orch, bound_responses)
        orch.model.invoke.side_effect = verdict_responses
        return orch

    def test_done_stops_after_one_round(self, make_orchestrator):
        orch = self._manual(
            make_orchestrator,
            [AIMessage(content="答案第一版")],
            [AIMessage(content="[DONE]")],
        )
        events = list(orch.run_stream("帮我查个事"))
        assert "答案第一版" in events[-1].content
        assert _humans(orch) == ["帮我查个事"]

    def test_continue_injects_next_instruction(self, make_orchestrator):
        orch = self._manual(
            make_orchestrator,
            [AIMessage(content="第一轮"), AIMessage(content="第二轮")],
            [
                AIMessage(content="[CONTINUE]\n缺少引用\n请补充引用即可"),
                AIMessage(content="[DONE]"),
            ],
        )
        events = list(orch.run_stream("写个调研报告"))
        verify_thinks = [
            e
            for e in events
            if e.step == StepType.THINK and e.metadata.get("phase") == "verify"
        ]
        assert len(verify_thinks) == 1
        assert "请补充引用即可" in verify_thinks[0].content
        assert "请补充引用即可" in _humans(orch)
        assert "第一轮第二轮" in events[-1].content

    def test_max_rounds_limits_iterations(self, make_orchestrator):
        orch, _ = make_orchestrator(
            use_manual_react=True, enable_verify=True, verify_max_rounds=1
        )
        _ready_bound_model(orch, [AIMessage(content="仅一轮")])
        orch.model.invoke.side_effect = [AIMessage(content="[CONTINUE]\n继续")]
        events = list(orch.run_stream("hi"))
        assert "仅一轮" in events[-1].content
        assert _humans(orch) == ["hi"]

    def test_unparseable_verdict_stops(self, make_orchestrator):
        orch = self._manual(
            make_orchestrator,
            [AIMessage(content="回复")],
            [AIMessage(content="lorem ipsum")],
        )
        events = list(orch.run_stream("hi"))
        assert "回复" in events[-1].content
        assert _humans(orch) == ["hi"]

    def test_verdict_model_error_is_safe(self, make_orchestrator):
        orch = self._manual(
            make_orchestrator, [AIMessage(content="回复")], RuntimeError("boom")
        )
        events = list(orch.run_stream("hi"))
        assert "回复" in events[-1].content
        assert "[模型调用失败]" not in events[-1].content

    def test_verify_with_plan_execute(self, make_orchestrator):
        orch, _ = make_orchestrator(strategy="plan_execute", enable_verify=True)
        orch.model.invoke.side_effect = [
            AIMessage(content="计划：1.分析 2.回答"),
            AIMessage(content="[DONE]"),
        ]
        bound = MagicMock()
        bound.invoke.side_effect = [AIMessage(content="执行完成")]
        orch.model.bind_tools.return_value = bound
        events = list(orch.run_stream("办件事"))
        plan_events = [e for e in events if e.metadata.get("plan")]
        assert len(plan_events) == 1
        assert "执行完成" in events[-1].content

    def test_verify_with_langgraph(self, make_orchestrator):
        script = [
            make_msg_chunk(AIMessageChunk(content="版本一", id="r1"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="版本一")]}}),
            make_msg_chunk(AIMessageChunk(content="版本二", id="r2"), "model"),
            make_update_chunk({"model": {"messages": [AIMessage(content="版本二")]}}),
        ]
        orch, _ = make_orchestrator(script=script, enable_verify=True)
        orch.model.invoke.side_effect = [
            AIMessage(content="[CONTINUE]\n需要更多\n再试一次"),
            AIMessage(content="[DONE]"),
        ]
        events = list(orch.run_stream("任务"))
        assert "版本一版本二" in events[-1].content
        assert "再试一次" in _humans(orch)

    def test_verify_flag_shows_in_config(self, make_orchestrator):
        orch, _ = make_orchestrator(enable_verify=True, verify_max_rounds=3)
        info = orch.get_config_info()
        assert info["verify_enabled"] is True
        assert info["verify_max_rounds"] == 3


class TestSetVerify:
    def test_toggle(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.set_verify(True) is True
        assert orch.get_config_info()["verify_enabled"] is True
        assert orch.set_verify(False) is False

    def test_max_rounds_sanitized(self, make_orchestrator):
        orch, _ = make_orchestrator()
        orch.set_verify(True, max_rounds=0)
        assert orch._verify_max_rounds >= 1
