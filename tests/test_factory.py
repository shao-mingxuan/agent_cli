"""create_orchestrator 的 MCP 自动发现测试。"""

from types import SimpleNamespace
from typing import ClassVar

import agent_cli.cli.factory as factory_mod
from agent_cli.cli.factory import create_orchestrator


class _FakeRegistry:
    def __init__(self, log: list):
        self.log = log

    def add_server(self, config):
        self.log.append(config.name)
        return 1


class _FakeOrchestrator:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        type(self).instances.append(kwargs)


def _patch(monkeypatch, log: list):
    monkeypatch.setattr(factory_mod, "MCPRegistry", lambda: _FakeRegistry(log))
    monkeypatch.setattr(factory_mod, "Orchestrator", _FakeOrchestrator)
    _FakeOrchestrator.instances = []


class TestMcpAutoLoad:
    def test_auto_load_when_default_config_exists(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "mcp.json").write_text('{"mcpServers":{}}')
        log: list = []
        _patch(monkeypatch, log)
        monkeypatch.setattr(
            factory_mod,
            "load_mcp_servers_from_config",
            lambda p: [SimpleNamespace(name="auto")],
        )
        create_orchestrator()
        assert log == ["auto"]
        mcp_registry = _FakeOrchestrator.instances[0]["mcp_registry"]
        assert mcp_registry is not None

    def test_explicit_config_skips_auto(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "mcp.json").write_text('{"mcpServers":{}}')
        log: list = []
        calls: list = []
        _patch(monkeypatch, log)

        def _fake_load(path):
            calls.append(path)
            return [SimpleNamespace(name="explicit")]

        monkeypatch.setattr(factory_mod, "load_mcp_servers_from_config", _fake_load)
        create_orchestrator(mcp_config="custom.json")
        assert calls == ["custom.json"]
        assert log == ["explicit"]

    def test_no_mcpjson_no_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        log: list = []
        _patch(monkeypatch, log)
        monkeypatch.setattr(factory_mod, "load_mcp_servers_from_config", lambda p: [])
        create_orchestrator()
        assert log == []
        assert _FakeOrchestrator.instances[0]["mcp_registry"] is None

    def test_bad_mcpjson_does_not_raise(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "mcp.json").write_text("{not-json")
        log: list = []
        _patch(monkeypatch, log)
        create_orchestrator()
        assert log == []
        assert _FakeOrchestrator.instances[0]["mcp_registry"] is None
        assert "MCP 配置文件加载失败" in capsys.readouterr().out
