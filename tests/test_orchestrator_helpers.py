"""L1 Orchestrator 纯辅助函数测试 (无需 mock)。"""

from agent_cli.agent.approval import (
    _SENSITIVE_PATTERNS,
    _SENSITIVE_WORDS,
    _TOOL_ERROR_MARKERS,
    MAX_TOOL_FAILURES,
    builtin_approval_callback,
    is_sensitive_tool,
    is_tool_error,
)


class TestIsToolError:
    def test_with_error_marker(self):
        assert is_tool_error("Error: something") is True

    def test_with_exception_marker(self):
        assert is_tool_error("Exception occurred") is True

    def test_with_traceback_marker(self):
        assert is_tool_error("traceback at line 5") is True

    def test_with_failed_marker(self):
        assert is_tool_error("operation failed") is True

    def test_with_mcp_error_marker(self):
        assert is_tool_error("mcp error: timeout") is True

    def test_with_normal_content(self):
        assert is_tool_error("success result") is False

    def test_with_empty_string(self):
        assert is_tool_error("") is False

    def test_with_none(self):
        assert is_tool_error(None) is False

    def test_case_insensitive(self):
        assert is_tool_error("ERROR: big") is True
        assert is_tool_error("FAILED") is True


class TestIsSensitiveTool:
    def test_file_keyword(self):
        assert is_sensitive_tool({"name": "read_file"}) is True

    def test_shell_keyword(self):
        assert is_sensitive_tool({"name": "execute_shell"}) is True

    def test_sql_keyword(self):
        assert is_sensitive_tool({"name": "query_db"}) is True

    def test_write_keyword(self):
        assert is_sensitive_tool({"name": "write_data"}) is True

    def test_non_sensitive_tool(self):
        assert is_sensitive_tool({"name": "calculate"}) is False

    def test_category_match(self):
        assert is_sensitive_tool({"name": "x", "category": "filesystem"}) is True

    def test_non_sensitive_category(self):
        assert (
            is_sensitive_tool({"name": "calculate", "category": "calculate"}) is False
        )

    def test_empty_name(self):
        assert is_sensitive_tool({"name": ""}) is False

    def test_empty_name_and_category(self):
        assert is_sensitive_tool({"name": "", "category": ""}) is False

    def test_combined_name_and_category(self):
        assert is_sensitive_tool({"name": "search", "category": "local"}) is False

    def test_word_match_no_false_positive_sh(self):
        assert is_sensitive_tool({"name": "hash"}) is False
        assert is_sensitive_tool({"name": "flash"}) is False
        assert is_sensitive_tool({"name": "finish"}) is False

    def test_word_match_no_false_positive_db(self):
        assert is_sensitive_tool({"name": "adblock"}) is False
        assert is_sensitive_tool({"name": "debug"}) is False

    def test_word_match_db_in_query_db(self):
        assert is_sensitive_tool({"name": "query_db"}) is True

    def test_word_match_shell_in_execute_shell(self):
        assert is_sensitive_tool({"name": "execute_shell"}) is True

    def test_word_match_file_in_read_file(self):
        assert is_sensitive_tool({"name": "read_file"}) is True

    def test_pattern_match_read_file(self):
        assert is_sensitive_tool({"name": "read_file"}) is True

    def test_pattern_match_fs_prefix(self):
        assert is_sensitive_tool({"name": "fs_list"}) is True

    def test_no_false_positive_code(self):
        assert is_sensitive_tool({"name": "encode"}) is False
        assert is_sensitive_tool({"name": "decode"}) is False
        assert is_sensitive_tool({"name": "barcode"}) is False

    def test_no_false_positive_run(self):
        assert is_sensitive_tool({"name": "runtime"}) is False
        assert is_sensitive_tool({"name": "runner"}) is False

    def test_exact_word_push_still_matches(self):
        assert is_sensitive_tool({"name": "git_push"}) is True
        assert is_sensitive_tool({"name": "push"}) is True


class TestBuiltinApprovalCallback:
    def test_non_sensitive_approved(self):
        result = builtin_approval_callback([{"sensitive": False}])
        assert result == [True]

    def test_sensitive_rejected(self):
        result = builtin_approval_callback([{"sensitive": True}])
        assert result == [False]

    def test_mixed(self):
        result = builtin_approval_callback([{"sensitive": False}, {"sensitive": True}])
        assert result == [True, False]

    def test_empty_list(self):
        assert builtin_approval_callback([]) == []

    def test_default_sensitive_false(self):
        result = builtin_approval_callback([{}])
        assert result == [True]


class TestConstants:
    def test_max_tool_failures_is_3(self):
        assert MAX_TOOL_FAILURES == 3

    def test_tool_error_markers_not_empty(self):
        assert "error" in _TOOL_ERROR_MARKERS
        assert "exception" in _TOOL_ERROR_MARKERS
        assert len(_TOOL_ERROR_MARKERS) >= 4

    def test_sensitive_words_not_empty(self):
        assert "file" in _SENSITIVE_WORDS
        assert "shell" in _SENSITIVE_WORDS
        assert len(_SENSITIVE_WORDS) >= 15

    def test_sensitive_patterns_not_empty(self):
        assert "read_file" in _SENSITIVE_PATTERNS
        assert "fs_" in _SENSITIVE_PATTERNS
        assert len(_SENSITIVE_PATTERNS) >= 3
