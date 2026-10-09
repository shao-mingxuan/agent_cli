"""L1 手动 ReAct 循环策略测试 (manual_react_loop)。"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from agent_cli.agent.types import StepType
from agent_cli.tools.registry import ToolInfo


def _ready_model(orch, responses):
    bound = MagicMock()
    bound.invoke.side_effect = responses
    orch.model.bind_tools.return_value = bound
    return bound


def _calc_tc(expr="1+1", tc_id="tc1"):
    return {
        "name": "calculate",
        "args": {"expression": expr},
        "id": tc_id,
        "type": "tool_call",
    }


class TestManualReactLoop:
    def test_simple_response(self, make_orchestrator):
        orch, _ = make_orchestrator(use_manual_react=True)
        _ready_model(orch, [AIMessage(content="大家好")])
        events = list(orch.run_stream("你好"))
        assert events[-1].step == StepType.RESPOND
        assert "大家好" in events[-1].content
        assert orch.memory.get_messages()[-1].content == "大家好"

    def test_tool_roundtrip(self, make_orchestrator):
        orch, _ = make_orchestrator(use_manual_react=True)
        first = AIMessage(content="", tool_calls=[_calc_tc()])
        second = AIMessage(content="结果是 2")
        _ready_model(orch, [first, second])
        events = list(orch.run_stream("1+1?"))
        steps = [e.step for e in events]
        assert StepType.THINK in steps
        assert StepType.ACT in steps
        assert "结果是 2" in events[-1].content
        tool_results = [
            m
            for m in orch.memory.get_messages()
            if isinstance(m, ToolMessage) and "计算结果" in m.content
        ]
        assert len(tool_results) == 1

    def test_sensitive_tool_triggers_approve(self, make_orchestrator):
        tool = MagicMock()
        tool.name = "write_secret"
        tool.invoke.return_value = "ok"
        info = ToolInfo(tool=tool, name="write_secret", source="custom")
        orch, _ = make_orchestrator(
            use_manual_react=True, extra_tool_infos=[info]
        )
        orch.set_approval_callback(lambda tcs: [True])
        tc = {
            "name": "write_secret",
            "args": {"data": "x"},
            "id": "tc1",
            "type": "tool_call",
        }
        _ready_model(orch, [AIMessage(content="", tool_calls=[tc]), AIMessage(content="done")])
        events = list(orch.run_stream("write it"))
        assert any(e.step == StepType.APPROVE for e in events)
        tool.invoke.assert_called_once_with({"data": "x"})

    def test_rejected_tool_injects_rejection(self, make_orchestrator):
        tool = MagicMock()
        tool.name = "write_secret"
        tool.invoke.return_value = "ok"
        info = ToolInfo(tool=tool, name="write_secret", source="custom")
        orch, _ = make_orchestrator(
            use_manual_react=True, extra_tool_infos=[info]
        )
        orch.set_approval_callback(lambda tcs: [False])
        tc = {
            "name": "write_secret",
            "args": {"data": "x"},
            "id": "tc1",
            "type": "tool_call",
        }
        _ready_model(orch, [AIMessage(content="", tool_calls=[tc]), AIMessage(content="ok")])
        events = list(orch.run_stream("write it"))
        rejected = [
            e
            for e in events
            if e.step == StepType.ACT and e.metadata.get("rejected")
        ]
        assert len(rejected) == 1
        assert rejected[0].content == "用户拒绝了此工具调用。"
        tool.invoke.assert_not_called()

    def test_abort_after_three_tool_failures(self, make_orchestrator):
        flaky = MagicMock()
        flaky.name = "flaky_tool"
        flaky.invoke.return_value = "an error occurred"
        info = ToolInfo(tool=flaky, name="flaky_tool", source="custom")
        orch, _ = make_orchestrator(
            use_manual_react=True, extra_tool_infos=[info]
        )
        tc = {
            "name": "flaky_tool",
            "args": {"cmd": "x"},
            "id": "tc1",
            "type": "tool_call",
        }
        failing = AIMessage(content="", tool_calls=[tc])
        _ready_model(orch, [failing, failing, failing])
        events = list(orch.run_stream("flaky"))
        assert "连续调用失败" in events[-1].content

    def test_model_error_becomes_respond(self, make_orchestrator):
        orch, _ = make_orchestrator(use_manual_react=True)
        bound = MagicMock()
        bound.invoke.side_effect = RuntimeError("bad api key")
        orch.model.bind_tools.return_value = bound
        events = list(orch.run_stream("hi"))
        assert "[模型调用失败]" in events[-1].content

    def test_manual_defaults_off(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.get_config_info()["react_loop"] == "langgraph"


class TestSetReactMode:
    def test_toggle(self, make_orchestrator):
        orch, _ = make_orchestrator()
        assert orch.set_react_mode(True) is True
        assert orch.get_config_info()["react_loop"] == "manual"
        assert orch.set_react_mode(False) is False
        assert orch.get_config_info()["react_loop"] == "langgraph"

    def test_none_keeps_current(self, make_orchestrator):
        orch, _ = make_orchestrator(use_manual_react=True)
        assert orch.set_react_mode(None) is True

    def test_run_stream_uses_manual_loop(self, make_orchestrator):
        orch, _ = make_orchestrator(use_manual_react=True)
        _ready_model(orch, [AIMessage(content="manual")])
        events = list(orch.run_stream("hi"))
        assert "manual" in events[-1].content
