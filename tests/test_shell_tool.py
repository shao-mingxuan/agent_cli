"""L3 shell 命令执行工具测试。"""

from agent_cli.tools.builtin.shell import run_command


class TestRunCommand:
    def test_success(self):
        out = run_command.invoke({"command": "echo hello"})
        assert "Exit Code: 0" in out
        assert "hello" in out

    def test_failure_exit_code(self):
        out = run_command.invoke({"command": "false"})
        assert "Exit Code: 1" in out

    def test_command_not_found(self):
        out = run_command.invoke({"command": "definitely_not_a_command_xyz"})
        assert "not found" in out

    def test_empty_command(self):
        out = run_command.invoke({"command": "   "})
        assert "Error" in out

    def test_quoted_args(self):
        out = run_command.invoke({"command": 'echo "a b"'})
        assert "a b" in out

    def test_shell_syntax_and(self):
        out = run_command.invoke({"command": "echo first && echo second"})
        assert "first" in out and "second" in out
        assert "Exit Code: 0" in out

    def test_shell_pipeline(self):
        out = run_command.invoke({"command": "echo hi | tr a-z A-Z"})
        assert "HI" in out

    def test_cwd(self, tmp_path):
        out = run_command.invoke({"command": "pwd", "cwd": str(tmp_path)})
        assert str(tmp_path) in out

    def test_timeout(self):
        out = run_command.invoke({"command": "sleep 3", "timeout": 1})
        assert "超过" in out or "中断" in out
