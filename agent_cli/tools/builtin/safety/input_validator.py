"""L3 工具 - 输入规范校验。"""

from langchain.tools import tool

from .backend import get_backend


@tool
def validate_input(text: str) -> str:
    """校验输入文本的规范性，检测空输入、过长输入、控制字符、SQL注入、Prompt注入等异常。

    Args:
        text: 待校验的文本内容
    """
    result = get_backend().validate_input(text)

    lines = [f"校验结果: {result.risk_level}"]
    if result.findings:
        lines.append(f"发现 {len(result.findings)} 个问题:")
        for f in result.findings:
            lines.append(f"  - 类型: {f['type']}, 详情: {f['value']}")
    lines.append(f"结论: {result.message}")
    return "\n".join(lines)
