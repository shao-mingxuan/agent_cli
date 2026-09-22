"""L6 中间件类型测试。"""
from agent_cli.middleware.types import GuardResult, AgentMiddleware


class TestGuardResult:
    def test_defaults(self):
        r = GuardResult()
        assert r.blocked is False
        assert r.risk_level == "safe"
        assert r.message == ""
        assert r.findings == []

    def test_blocked_true(self):
        r = GuardResult(blocked=True)
        assert r.blocked is True

    def test_danger_level(self):
        r = GuardResult(risk_level="danger")
        assert r.risk_level == "danger"

    def test_findings_independent(self):
        r1 = GuardResult()
        r2 = GuardResult()
        r1.findings.append({"type": "x"})
        assert r2.findings == []

    def test_with_findings(self):
        r = GuardResult(findings=[{"type": "sql_injection"}])
        assert len(r.findings) == 1

    def test_with_message(self):
        r = GuardResult(message="blocked by safety guard")
        assert r.message == "blocked by safety guard"
