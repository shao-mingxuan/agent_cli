"""L6 中间件 - 敏感信息脱敏。"""

import re
from typing import Iterable

from ..types import GuardResult

_TYPES = {
    "phone": (r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]"),
    "id_card": (r"(?<!\d)[1-9]\d{16}[\dXx](?!\d)", "[身份证]"),
    "email": (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱]"),
    "ip": (r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", "[IP]"),
}


def redact_text(text: str, types: Iterable[str] | None = None) -> tuple[str, list[dict]]:
    """把文本中的手机号/身份证/邮箱/IP 替换为占位符。

    Returns:
        (脱敏后的文本, findings 列表)。findings 每项含 type/count。
    """
    if not text:
        return text, []
    names = list(types) if types is not None else list(_TYPES)
    findings: list[dict] = []
    result = text
    for name in names:
        pattern, replacement = _TYPES[name]
        count = len(re.findall(pattern, result))
        if count:
            findings.append({"type": name, "count": count})
            result = re.sub(pattern, replacement, result)
    return result, findings


class RedactionMiddleware:
    """输出脱敏检测：模型回复含敏感信息时给出 warning。

    仅做检测与提示，不修改回复内容（协议不支持改写文本）。
    如需脱敏展示，可调用 redact_text() 对最终文本做替换。
    """

    def before_run(self, user_input: str) -> GuardResult:
        return GuardResult()

    def after_run(self, user_input: str, response: str) -> GuardResult:
        _, findings = redact_text(response)
        if not findings:
            return GuardResult()
        labels = ", ".join(f"{f['type']}x{f['count']}" for f in findings)
        return GuardResult(
            risk_level="warning",
            message=f"回复包含敏感信息（{labels}），建议脱敏后再展示。",
            findings=findings,
        )
