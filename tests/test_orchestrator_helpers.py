"""L1 Orchestrator 纯辅助函数测试 (无需 mock)。"""
from agent_cli.agent.orchestrator import (
    MAX_TOOL_FAILURES,
    _TOOL_ERROR_MARKERS,
    _SENSITIVE_KEYWORDS,
    _is_tool_error,
    Orchestrator,
)


def _check_sensitive(tool_call):
    """Orchestrator._is_sensitive_tool 是实例方法，通过 __new__ 跳过 __init__ 绑定。"""
    orch = Orchestrator.__new__(Orchestrator)
    return orch._is_sensitive_tool(tool_call)


class TestIsToolError:
    def test_with_error_marker(self):
        assert _is_tool_error("Error: something") is True

    def test_with_exception_marker(self):
        assert _is_tool_error("Exception occurred") is True

    def test_with_traceback_marker(self):
        assert _is_tool_error("traceback at line 5") is True

    def test_with_failed_marker(self):
        assert _is_tool_error("operation failed") is True

    def test_with_mcp_error_marker(self):
        assert _is_tool_error("mcp error: timeout") is True

    def test_with_normal_content(self):
        assert _is_tool_error("success result") is False

    def test_with_empty_string(self):
        assert _is_tool_error("") is False

    def test_with_none(self):
        assert _is_tool_error(None) is False

    def test_case_insensitive(self):
        assert _is_tool_error("ERROR: big") is True
        assert _is_tool_error("FAILED") is True


class TestIsSensitiveTool:
    def test_file_keyword(self):
        assert _check_sensitive({"name": "read_file"}) is True

    def test_shell_keyword(self):
        assert _check_sensitive({"name": "execute_shell"}) is True

    def test_sql_keyword(self):
        assert _check_sensitive({"name": "query_db"}) is True

    def test_write_keyword(self):
        assert _check_sensitive({"name": "write_data"}) is True

    def test_non_sensitive_tool(self):
        assert _check_sensitive({"name": "calculate"}) is False

    def test_category_match(self):
        assert _check_sensitive({"name": "x", "category": "filesystem"}) is True

    def test_non_sensitive_category(self):
        assert _check_sensitive({"name": "calculate", "category": "calculate"}) is False

    def test_empty_name(self):
        assert _check_sensitive({"name": ""}) is False

    def test_empty_name_and_category(self):
        assert _check_sensitive({"name": "", "category": ""}) is False

    def test_combined_name_and_category(self):
        assert _check_sensitive({"name": "search", "category": "local"}) is True

    def test_substring_match_sh_in_push(self):
        assert _check_sensitive({"name": "push"}) is True

    def test_substring_match_db_in_adblock(self):
        assert _check_sensitive({"name": "adblock"}) is True


class TestBuiltinApprovalCallback:
    def test_non_sensitive_approved(self):
        result = Orchestrator._builtin_approval_callback([{"sensitive": False}])
        assert result == [True]

    def test_sensitive_rejected(self):
        result = Orchestrator._builtin_approval_callback([{"sensitive": True}])
        assert result == [False]

    def test_mixed(self):
        result = Orchestrator._builtin_approval_callback(
            [{"sensitive": False}, {"sensitive": True}]
        )
        assert result == [True, False]

    def test_empty_list(self):
        assert Orchestrator._builtin_approval_callback([]) == []

    def test_default_sensitive_false(self):
        result = Orchestrator._builtin_approval_callback([{}])
        assert result == [True]


class TestConstants:
    def test_max_tool_failures_is_3(self):
        assert MAX_TOOL_FAILURES == 3

    def test_tool_error_markers_not_empty(self):
        assert "error" in _TOOL_ERROR_MARKERS
        assert "exception" in _TOOL_ERROR_MARKERS
        assert len(_TOOL_ERROR_MARKERS) >= 4

    def test_sensitive_keywords_not_empty(self):
        assert "file" in _SENSITIVE_KEYWORDS
        assert "shell" in _SENSITIVE_KEYWORDS
        assert len(_SENSITIVE_KEYWORDS) >= 20
