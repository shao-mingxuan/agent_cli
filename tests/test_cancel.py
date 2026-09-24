"""L0 CLI - ESC 打断功能测试。"""

import io
import os
import unittest.mock as mock

from rich.console import Console

from agent_cli.agent.types import AgentEvent, StepType
from agent_cli.cli import display as display_mod
from agent_cli.cli.cancel import CancelController


def _event(step, content="", metadata=None):
    return AgentEvent(step=step, content=content, metadata=metadata or {})


class _StubAgent:
    """仅用于 handle_events 测试的假 agent。"""

    def __init__(self, events):
        self._events = list(events)

    def run_stream(self, user_input):
        for e in self._events:
            yield e


class _FakeDisplay:
    """记录调用轨迹的假展示对象。"""

    def __init__(self):
        self.tokens = []
        self.finished = False
        self.aborted = False
        self.saw_thinking = False

    def start_thinking(self):
        self.saw_thinking = True

    def stop_spinner(self):
        pass

    def append_token(self, token):
        self.tokens.append(token)

    def finish(self):
        self.finished = True
        return "".join(self.tokens)

    def abort(self):
        self.aborted = True


class _PreCancelledController:
    """is_cancelled 始终保持 True 的控制器。"""

    def start(self):
        pass

    def stop(self):
        pass

    def is_cancelled(self):
        return True


class _CancelAfterNController:
    """前 n 次返回 False，之后返回 True。"""

    def __init__(self, n):
        self._n = n
        self._i = 0

    def start(self):
        pass

    def stop(self):
        pass

    def is_cancelled(self):
        self._i += 1
        return self._i > self._n


def _run(agent, controller):
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=False)
    display = _FakeDisplay()
    with mock.patch.object(display_mod, "console", c):
        with mock.patch.object(display_mod, "CancelController", lambda: controller):
            with mock.patch.object(display_mod, "TypewriterDisplay", lambda: display):
                display_mod.handle_events(agent, "hi")
    return buf.getvalue(), display


class TestCancelController:
    def test_init_not_cancelled(self):
        c = CancelController()
        assert not c.is_cancelled()
        assert not c.active

    def test_manual_cancel(self):
        c = CancelController()
        c.cancel()
        assert c.is_cancelled()

    def test_stop_after_manual_cancel_idempotent(self):
        c = CancelController()
        c._cancel.set()
        c.stop()
        c.stop()
        assert c.is_cancelled()

    def test_not_active_without_tty(self):
        c = CancelController()
        c.start()
        assert not c.active
        c.stop()

    def _run_listen(self, monkeypatch, key: bytes):
        import agent_cli.cli.cancel as cancel_mod

        c = CancelController()
        r, w = os.pipe()
        c._fd = r
        interrupted = []
        calls = {"n": 0}

        def fake_select(rlist, wlist, xlist, timeout):
            calls["n"] += 1
            return (rlist, [], []) if calls["n"] == 1 else ([], [], [])

        monkeypatch.setattr(
            cancel_mod, "interrupt_main", lambda: interrupted.append(True)
        )
        monkeypatch.setattr(cancel_mod.select, "select", fake_select)
        try:
            os.write(w, key)
            c._stop.clear()
            c._listen()
        finally:
            os.close(r)
            os.close(w)
        return c, interrupted

    def test_esc_triggers_cancel_without_interrupt(self, monkeypatch):
        c, interrupted = self._run_listen(monkeypatch, b"\x1b")
        assert c.is_cancelled()
        assert interrupted == []

    def test_ctrl_c_triggers_cancel_and_interrupt(self, monkeypatch):
        c, interrupted = self._run_listen(monkeypatch, b"\x03")
        assert c.is_cancelled()
        assert interrupted == [True]


class TestHandleEventsInterrupt:
    def test_preserved_order_without_cancel(self):
        agent = _StubAgent(
            [
                _event(StepType.THINKING),
                _event(StepType.TOKEN, "你好"),
                _event(StepType.RESPOND, "world"),
            ]
        )
        out, display = _run(agent, _CancelAfterNController(99))
        assert display.saw_thinking
        assert "".join(display.tokens) == "你好"
        assert display.finished
        assert not display.aborted
        assert "Interrupted" not in out

    def test_breaks_immediately_when_pre_cancelled(self):
        agent = _StubAgent(
            [
                _event(StepType.THINKING),
                _event(StepType.TOKEN, "不该输出"),
                _event(StepType.RESPOND, "done"),
            ]
        )
        out, display = _run(agent, _PreCancelledController())
        assert not display.saw_thinking
        assert display.tokens == []
        assert not display.finished
        assert display.aborted
        assert "Interrupted" in out

    def test_breaks_after_first_token(self):
        agent = _StubAgent(
            [
                _event(StepType.THINKING),
                _event(StepType.TOKEN, "aa"),
                _event(StepType.TOKEN, "bb"),
                _event(StepType.RESPOND, "done"),
            ]
        )
        out, display = _run(agent, _CancelAfterNController(2))
        assert "".join(display.tokens) == "aa"
        assert not display.finished
        assert display.aborted
        assert "Interrupted" in out


class TestTypewriterAbort:
    def test_abort_clears_buffer(self):
        d = display_mod.TypewriterDisplay()
        d.append_token("abc")
        d.abort()
        assert d._buffer == ""
        assert d._live is None
        assert d._status is None
