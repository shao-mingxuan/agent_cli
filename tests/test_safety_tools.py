"""L3 安全工具测试 - @tool 包装的安全检测函数。"""
import os
from unittest.mock import patch

from agent_cli.tools.builtin.safety.pii_detector import detect_pii
from agent_cli.tools.builtin.safety.content_moderator import detect_violation
from agent_cli.tools.builtin.safety.file_scanner import scan_file
from agent_cli.tools.builtin.safety.input_validator import validate_input


class TestDetectPiiTool:
    def test_safe_text(self):
        result = detect_pii.invoke({"text": "hello world"})
        assert "safe" in result

    def test_with_phone(self):
        result = detect_pii.invoke({"text": "call 13812345678"})
        assert "warning" in result
        assert "phone" in result

    def test_output_format(self):
        result = detect_pii.invoke({"text": "hello"})
        assert "风险等级" in result
        assert "结论" in result

    def test_func_direct_call(self):
        result = detect_pii.func("hello")
        assert "safe" in result

    def test_with_email(self):
        result = detect_pii.invoke({"text": "contact user@example.com"})
        assert "email" in result


class TestDetectViolationTool:
    def test_safe_text(self):
        result = detect_violation.invoke({"text": "hello"})
        assert "safe" in result

    def test_with_profanity(self):
        result = detect_violation.invoke({"text": "你是笨蛋"})
        assert "danger" in result

    def test_output_format(self):
        result = detect_violation.invoke({"text": "你是笨蛋"})
        assert "风险等级" in result
        assert "命中" in result

    def test_func_direct_call(self):
        result = detect_violation.func("hello")
        assert "safe" in result


class TestValidateInputTool:
    def test_normal(self):
        result = validate_input.invoke({"text": "hello"})
        assert "safe" in result

    def test_sql_injection(self):
        result = validate_input.invoke({"text": "' OR '1'='1"})
        assert "danger" in result

    def test_empty(self):
        result = validate_input.invoke({"text": ""})
        assert "warning" in result
        assert "empty" in result

    def test_output_uses_check_label(self):
        result = validate_input.invoke({"text": "hello"})
        assert "校验结果" in result


class TestScanFileTool:
    def test_nonexistent(self, tmp_path):
        result = scan_file.invoke({"filepath": str(tmp_path / "nonexistent")})
        assert "文件不存在" in result

    def test_directory(self, tmp_path):
        result = scan_file.invoke({"filepath": str(tmp_path)})
        assert "目录" in result

    def test_safe_content(self, tmp_path):
        f = tmp_path / "safe.txt"
        f.write_text("hello world")
        result = scan_file.invoke({"filepath": str(f)})
        assert "safe" in result

    def test_with_sensitive_path(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("KEY=value")
        result = scan_file.invoke({"filepath": str(f)})
        assert "danger" in result

    def test_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * (1024 * 1024 + 1))
        result = scan_file.invoke({"filepath": str(f)})
        assert "文件过大" in result

    def test_with_secret_content(self, tmp_path):
        f = tmp_path / "config.txt"
        f.write_text("password=secret123")
        result = scan_file.invoke({"filepath": str(f)})
        assert "danger" in result
