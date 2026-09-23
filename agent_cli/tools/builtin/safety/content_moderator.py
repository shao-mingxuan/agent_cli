"""L3 工具 - 违规词/敏感内容检测。"""

from langchain.tools import tool

from .backend import get_backend


@tool
def detect_violation(text: str) -> str:
    """检测文本中是否包含违规词、敏感词或辱骂内容。

    Args:
        text: 待检测的文本内容
    """
    result = get_backend().detect_violation(text)

    lines = [f"风险等级: {result.risk_level}"]
    if result.findings:
        lines.append(f"命中 {len(result.findings)} 个违规词:")
        for f in result.findings:
            lines.append(f"  - {f['value']}")
    lines.append(f"结论: {result.message}")
    return "\n".join(lines)
