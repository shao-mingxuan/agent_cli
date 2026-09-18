"""L3 工具 - PII 个人隐私信息检测。"""
from langchain.tools import tool

from .backend import get_backend


@tool
def detect_pii(text: str) -> str:
    """检测文本中的个人隐私信息（PII），如手机号、身份证号、邮箱、银行卡号、IP地址等，并返回脱敏结果。

    Args:
        text: 待检测的文本内容
    """
    result = get_backend().detect_pii(text)

    lines = [f"风险等级: {result.risk_level}"]
    if result.findings:
        lines.append(f"检测到 {len(result.findings)} 个 PII:")
        for f in result.findings:
            lines.append(f"  - 类型: {f['type']}, 脱敏值: {f['value']}")
    lines.append(f"结论: {result.message}")
    return "\n".join(lines)
