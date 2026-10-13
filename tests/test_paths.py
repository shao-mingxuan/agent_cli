"""全局资源路径解析测试（AGENT_HOME / XDG / 项目根 / cwd 优先级）。"""

import os

from agent_cli import paths


class TestFindResource:
    def test_user_config_priority(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        user_skills = tmp_path / "xdg" / "agent_cli" / "skills"
        user_skills.mkdir(parents=True)
        assert paths.find_resource("skills") == user_skills

    def test_agent_home_overrides_config(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        xdg = tmp_path / "xdg"
        home.mkdir()
        (home / "mcp.json").write_text("{}")
        (xdg / "agent_cli").mkdir(parents=True)
        (xdg / "agent_cli" / "mcp.json").write_text("{}")
        monkeypatch.setenv("AGENT_HOME", str(home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        assert paths.find_resource("mcp.json") == home / "mcp.json"

    def test_project_root_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        assert paths.find_resource("skills") == paths.PROJECT_ROOT / "skills"

    def test_cwd_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo.txt").write_text("x")
        assert paths.find_resource("foo.txt") == tmp_path / "foo.txt"

    def test_missing_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        monkeypatch.chdir(tmp_path)
        assert paths.find_resource("missing.txt") is None


class TestResolveDefault:
    def test_skills_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        assert paths.skills_dir() == str(paths.PROJECT_ROOT / "skills")

    def test_plugins_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        assert paths.plugins_dir() == str(paths.PROJECT_ROOT / "plugins")

    def test_mcp_config_path_project_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        assert paths.mcp_config_path() == str(paths.PROJECT_ROOT / "mcp.json")

    def test_mcp_config_path_user_first(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg"
        (xdg / "agent_cli").mkdir(parents=True)
        (xdg / "agent_cli" / "mcp.json").write_text("{}")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.delenv("AGENT_HOME", raising=False)
        assert paths.mcp_config_path() == str(xdg / "agent_cli" / "mcp.json")


class TestLoadEnv:
    _KEY = "PATHS_TEST_VAR"

    def _clear_env(self, monkeypatch):
        monkeypatch.delenv(self._KEY, raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("AGENT_HOME", raising=False)

    def test_agent_home_wins(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)
        home = tmp_path / "home"
        xdg = tmp_path / "xdg"
        home.mkdir()
        (home / ".env").write_text(f"{self._KEY}=home")
        (xdg / "agent_cli").mkdir(parents=True)
        (xdg / "agent_cli" / ".env").write_text(f"{self._KEY}=xdg")
        (tmp_path / "cwd").mkdir()
        monkeypatch.setenv("AGENT_HOME", str(home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.chdir(str(tmp_path / "cwd"))
        paths.load_env()
        assert os.environ.get(self._KEY) == "home"

    def test_xdg_wins_over_cwd(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)
        xdg = tmp_path / "xdg"
        (xdg / "agent_cli").mkdir(parents=True)
        (xdg / "agent_cli" / ".env").write_text(f"{self._KEY}=xdg")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.chdir(tmp_path)
        paths.load_env()
        assert os.environ.get(self._KEY) == "xdg"

    def test_load_does_not_override_existing(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)
        xdg = tmp_path / "xdg"
        (xdg / "agent_cli").mkdir(parents=True)
        (xdg / "agent_cli" / ".env").write_text(f"{self._KEY}=xdg")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.setenv(self._KEY, "process")
        paths.load_env()
        assert os.environ.get(self._KEY) == "process"
