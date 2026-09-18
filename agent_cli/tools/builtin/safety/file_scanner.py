"""L3 工具 - 文件隐私/敏感信息扫描。"""
import os

from langchain.tools import tool

from .backend import get_backend


_MAX_FILE_SIZE = 1024 * 1024


@tool
def scan_file(filepath: str) -> str:
    """扫描指定文件是否包含敏感信息，如密钥、密码、Token、PII 等。支持检测 .env、密钥文件等敏感文件。

    Args:
        filepath: 待扫描的文件路径
    """
    if not os.path.exists(filepath):
        return f"错误: 文件不存在: {filepath}"
    if os.path.isdir(filepath):
        return f"错误: 路径是目录，非文件: {filepath}"

    size = os.path.getsize(filepath)
    if size > _MAX_FILE_SIZE:
        return f"错误: 文件过大 ({size} bytes)，最大支持 {_MAX_FILE_SIZE} bytes"

    try:
        with open(filepath, "r", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"错误: 读取文件失败: {e}"

    result = get_backend().scan_file(filepath, content)

    lines = [f"风险等级: {result.risk_level}"]
    if result.findings:
        lines.append(f"检测到 {len(result.findings)} 个敏感项:")
        for fnd in result.findings:
            lines.append(f"  - 类型: {fnd['type']}, 值: {fnd['value']}")
    lines.append(f"结论: {result.message}")
    return "\n".join(lines)
