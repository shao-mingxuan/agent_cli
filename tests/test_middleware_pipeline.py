"""L6 MiddlewarePipeline 测试。"""

from agent_cli.middleware.pipeline import MiddlewarePipeline, create_default_pipeline
from agent_cli.middleware.types import GuardResult


class FakeMiddleware:
    def __init__(self, before=None, after=None):
        self._before = before or GuardResult()
        self._after = after or GuardResult()

    def before_run(self, user_input):
        return self._before

    def after_run(self, user_input, response):
        return self._after


class TestEmptyPipeline:
    def test_empty_before_run(self):
        r = MiddlewarePipeline().before_run("x")
        assert r.blocked is False
        assert r.risk_level == "safe"
        assert r.findings == []

    def test_empty_after_run(self):
        r = MiddlewarePipeline().after_run("x", "y")
        assert r.blocked is False
        assert r.risk_level == "safe"
        assert r.findings == []


class TestSingleMiddleware:
    def test_safe_middleware(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult()))
        r = p.before_run("x")
        assert r.blocked is False
        assert r.risk_level == "safe"

    def test_blocked_middleware(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(blocked=True, risk_level="danger")))
        r = p.before_run("x")
        assert r.blocked is True
        assert r.risk_level == "danger"

    def test_warning_middleware(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(risk_level="warning")))
        r = p.before_run("x")
        assert r.risk_level == "warning"

    def test_with_findings(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(findings=[{"type": "x"}])))
        r = p.before_run("x")
        assert len(r.findings) == 1

    def test_with_message(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(message="blocked")))
        r = p.before_run("x")
        assert r.message == "blocked"


class TestMultipleMiddlewares:
    def test_all_safe(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult()))
        p.add(FakeMiddleware(before=GuardResult()))
        r = p.before_run("x")
        assert r.risk_level == "safe"
        assert r.blocked is False

    def test_one_blocks_all_blocked(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult()))
        p.add(FakeMiddleware(before=GuardResult(blocked=True, risk_level="danger")))
        r = p.before_run("x")
        assert r.blocked is True

    def test_max_risk_danger_wins(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(risk_level="warning")))
        p.add(FakeMiddleware(before=GuardResult(risk_level="danger")))
        r = p.before_run("x")
        assert r.risk_level == "danger"

    def test_max_risk_warning_when_no_danger(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(risk_level="safe")))
        p.add(FakeMiddleware(before=GuardResult(risk_level="warning")))
        r = p.before_run("x")
        assert r.risk_level == "warning"

    def test_findings_concatenated(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(findings=[{"type": "a"}])))
        p.add(FakeMiddleware(before=GuardResult(findings=[{"type": "b"}])))
        r = p.before_run("x")
        assert len(r.findings) == 2

    def test_messages_joined_with_semicolon(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(before=GuardResult(message="msg1")))
        p.add(FakeMiddleware(before=GuardResult(message="msg2")))
        r = p.before_run("x")
        assert "msg1" in r.message
        assert "msg2" in r.message
        assert "; " in r.message

    def test_after_run_aggregation(self):
        p = MiddlewarePipeline()
        p.add(FakeMiddleware(after=GuardResult(findings=[{"type": "a"}])))
        p.add(FakeMiddleware(after=GuardResult(blocked=True, risk_level="danger")))
        r = p.after_run("x", "y")
        assert r.blocked is True
        assert r.risk_level == "danger"
        assert len(r.findings) == 1


class TestCreateDefaultPipeline:
    def test_has_safety_guard(self):
        p = create_default_pipeline()
        assert len(p._middlewares) == 1
        from agent_cli.middleware.builtin.safety_guard import SafetyGuardMiddleware

        assert isinstance(p._middlewares[0], SafetyGuardMiddleware)
