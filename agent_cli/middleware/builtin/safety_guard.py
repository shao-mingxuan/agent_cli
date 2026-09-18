"""L6 中间件 - 双层安全检测中间件。"""
from ...tools.builtin.safety.backend import get_backend
from ..types import GuardResult


class SafetyGuardMiddleware:
    """安全检测中间件：前置检测输入，后置检测模型回复。"""

    def before_run(self, user_input: str) -> GuardResult:
        """前置检测：对用户输入做违规词 + 输入规范校验。"""
        backend = get_backend()
        findings: list[dict] = []
        messages: list[str] = []
        max_risk = "safe"
        blocked = False

        violation = backend.detect_violation(user_input)
        if violation.findings:
            findings.extend(violation.findings)
            messages.append(violation.message)
            if violation.risk_level == "danger":
                max_risk = "danger"
                blocked = True

        validation = backend.validate_input(user_input)
        if validation.findings:
            findings.extend(validation.findings)
            messages.append(validation.message)
            if validation.risk_level == "danger":
                max_risk = "danger"
                blocked = True
            elif validation.risk_level == "warning" and max_risk != "danger":
                max_risk = "warning"

        return GuardResult(
            blocked=blocked,
            risk_level=max_risk,
            message=" | ".join(messages) if messages else "",
            findings=findings,
        )

    def after_run(self, user_input: str, response: str) -> GuardResult:
        """后置检测：对模型回复做 PII 泄露检测。"""
        backend = get_backend()
        findings: list[dict] = []
        messages: list[str] = []
        max_risk = "safe"
        blocked = False

        pii = backend.detect_pii(response)
        if pii.findings:
            findings.extend(pii.findings)
            messages.append(f"回复中检测到 PII 泄露: {pii.message}")
            max_risk = "warning"

        violation = backend.detect_violation(response)
        if violation.findings:
            findings.extend(violation.findings)
            messages.append(f"回复中检测到违规内容: {violation.message}")
            max_risk = "danger"
            blocked = True

        return GuardResult(
            blocked=blocked,
            risk_level=max_risk,
            message=" | ".join(messages) if messages else "",
            findings=findings,
        )
