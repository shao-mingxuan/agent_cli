"""L6 中间件 - 执行 pipeline。"""

from .types import AgentMiddleware, GuardResult


class MiddlewarePipeline:
    """中间件管道：按顺序执行所有中间件的前置/后置检测。"""

    def __init__(self):
        self._middlewares: list[AgentMiddleware] = []

    def add(self, middleware: AgentMiddleware) -> None:
        self._middlewares.append(middleware)

    def before_run(self, user_input: str) -> GuardResult:
        """依次执行所有前置检测，任一 blocked 则拦截。"""
        all_findings: list[dict] = []
        max_risk = "safe"
        blocked = False
        messages: list[str] = []

        for mw in self._middlewares:
            result = mw.before_run(user_input)
            if result.findings:
                all_findings.extend(result.findings)
            if result.blocked:
                blocked = True
            if result.risk_level == "danger":
                max_risk = "danger"
            elif result.risk_level == "warning" and max_risk != "danger":
                max_risk = "warning"
            if result.message:
                messages.append(result.message)

        return GuardResult(
            blocked=blocked,
            risk_level=max_risk,
            message="; ".join(messages) if messages else "",
            findings=all_findings,
        )

    def after_run(self, user_input: str, response: str) -> GuardResult:
        """依次执行所有后置检测，对模型回复做安全检查。"""
        all_findings: list[dict] = []
        max_risk = "safe"
        blocked = False
        messages: list[str] = []

        for mw in self._middlewares:
            result = mw.after_run(user_input, response)
            if result.findings:
                all_findings.extend(result.findings)
            if result.blocked:
                blocked = True
            if result.risk_level == "danger":
                max_risk = "danger"
            elif result.risk_level == "warning" and max_risk != "danger":
                max_risk = "warning"
            if result.message:
                messages.append(result.message)

        return GuardResult(
            blocked=blocked,
            risk_level=max_risk,
            message="; ".join(messages) if messages else "",
            findings=all_findings,
        )


def create_default_pipeline() -> MiddlewarePipeline:
    """创建带有默认中间件的管道（仅含安全检测）。"""
    from .builtin.safety_guard import SafetyGuardMiddleware

    pipeline = MiddlewarePipeline()
    pipeline.add(SafetyGuardMiddleware())
    return pipeline


def create_full_pipeline() -> MiddlewarePipeline:
    """创建完整中间件管道：安全 + 费用 + 限流 + 脱敏 + 日志 + 追踪。

    相比默认管道增加了生产环境常用能力。限流默认 60 次/分钟，
    可在启用前按需调整各中间件参数。
    """
    from .builtin.cost_guard import CostGuardMiddleware
    from .builtin.logging import LoggingMiddleware
    from .builtin.rate_limit import RateLimitMiddleware
    from .builtin.redaction import RedactionMiddleware
    from .builtin.safety_guard import SafetyGuardMiddleware
    from .builtin.tracing import TracingMiddleware

    pipeline = MiddlewarePipeline()
    pipeline.add(SafetyGuardMiddleware())
    pipeline.add(CostGuardMiddleware())
    pipeline.add(RateLimitMiddleware())
    pipeline.add(RedactionMiddleware())
    pipeline.add(LoggingMiddleware())
    pipeline.add(TracingMiddleware())
    return pipeline
