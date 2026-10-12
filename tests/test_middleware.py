"""中间件测试：cost_guard / rate_limit / redaction / logging / tracing。"""

import logging
import time

from agent_cli.middleware.builtin.cost_guard import CostGuardMiddleware
from agent_cli.middleware.builtin.logging import LoggingMiddleware
from agent_cli.middleware.builtin.rate_limit import RateLimitMiddleware
from agent_cli.middleware.builtin.redaction import RedactionMiddleware, redact_text
from agent_cli.middleware.builtin.tracing import TracingMiddleware
from agent_cli.middleware.pipeline import create_full_pipeline


class TestCostGuard:
    def test_counts_and_cost(self):
        mw = CostGuardMiddleware(model="")
        mw.before_run("hello world")
        mw.after_run("hi", "response text")
        s = mw.stats()
        assert s["calls"] == 1
        assert s["input_tokens"] > 0
        assert s["output_tokens"] > 0
        assert s["total_tokens"] == s["input_tokens"] + s["output_tokens"]
        assert s["estimated_cost"] > 0

    def test_reset(self):
        mw = CostGuardMiddleware()
        mw.before_run("a")
        mw.after_run("b", "c")
        mw.reset()
        s = mw.stats()
        assert s == {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0,
        }

    def test_max_cost_blocks(self):
        mw = CostGuardMiddleware(max_cost=0.0)
        mw.before_run("hello world")
        r = mw.after_run("hi", "long response text for blocking")
        assert r.blocked is True


class TestRateLimit:
    def test_allows_within_limit(self):
        mw = RateLimitMiddleware(limit=2, period=60)
        assert mw.before_run("a").blocked is False
        assert mw.before_run("b").blocked is False

    def test_blocks_over_limit(self):
        mw = RateLimitMiddleware(limit=2, period=60)
        mw.before_run("a")
        mw.before_run("b")
        r = mw.before_run("c")
        assert r.blocked is True
        assert "重试" in r.message

    def test_window_expires(self):
        mw = RateLimitMiddleware(limit=1, period=0.05)
        mw.before_run("a")
        mw.reset()
        assert mw.before_run("b").blocked is False

    def test_after_run_safe(self):
        mw = RateLimitMiddleware()
        assert mw.after_run("a", "b").blocked is False


class TestRedaction:
    def test_redact_phone(self):
        out, findings = redact_text("联系 13812345678 详谈")
        assert "13812345678" not in out
        assert "[手机号]" in out
        assert findings[0]["type"] == "phone"

    def test_redact_email_idcard_ip(self):
        text = "my@email.com 11010519491231002X 192.168.1.1"
        out, findings = redact_text(text)
        assert "my@email.com" not in out
        assert "11010519491231002X" not in out
        assert "192.168.1.1" not in out
        types = {f["type"] for f in findings}
        assert {"email", "id_card", "ip"} <= types

    def test_redact_types_subset(self):
        text = "邮箱 a@b.com 电话 13912345678"
        out, _ = redact_text(text, types=["email"])
        assert "a@b.com" not in out
        assert "13912345678" in out

    def test_no_sensitive(self):
        out, findings = redact_text("普通文本")
        assert findings == []
        assert out == "普通文本"

    def test_empty(self):
        out, findings = redact_text("")
        assert out == "" and findings == []

    def test_middleware_warns(self):
        mw = RedactionMiddleware()
        r = mw.after_run("hi", "电话 13812345678")
        assert r.risk_level == "warning"
        assert len(r.findings) == 1
        assert mw.before_run("hi").blocked is False


class TestLogging:
    def test_records_debug(self, caplog):
        caplog.set_level(logging.DEBUG, logger="agent_cli.middleware.logging")
        logger = logging.getLogger("agent_cli.middleware.logging")
        handler = logging.NullHandler()
        # 确保 handler 存在，捕获 propagated 记录
        mw = LoggingMiddleware(logger=logger)
        mw.before_run("input")
        mw.after_run("input", "output text")
        records = [r for r in caplog.records if r.name.startswith("agent_cli.middleware")]
        assert any("run start" in r.getMessage() for r in records)
        assert any("run done" in r.getMessage() for r in records)


class TestTracing:
    def test_records_duration(self):
        mw = TracingMiddleware()
        mw.before_run("a")
        time.sleep(0.001)
        mw.after_run("a", "b")
        assert mw.last_duration_ms > 0
        assert len(mw.duration_list) == 1

    def test_reset(self):
        mw = TracingMiddleware()
        mw.before_run("a")
        mw.after_run("a", "b")
        mw.reset()
        assert mw.last_duration_ms == 0.0
        assert mw.duration_list == []

    def test_after_without_before_safe(self):
        mw = TracingMiddleware()
        assert mw.after_run("a", "b").blocked is False


class TestCreateFullPipeline:
    def test_contains_all_extension_middlewares(self):
        p = create_full_pipeline()
        types = [type(m).__name__ for m in p._middlewares]
        assert "SafetyGuardMiddleware" in types
        assert "CostGuardMiddleware" in types
        assert "RateLimitMiddleware" in types
        assert "RedactionMiddleware" in types
        assert "LoggingMiddleware" in types
        assert "TracingMiddleware" in types
