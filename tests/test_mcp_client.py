"""L3 MCP Client 测试 - 配置解析 + 客户端状态。"""

import json

import pytest

from agent_cli.mcp.client import (
    MCPClient,
    MCPServerConfig,
    load_mcp_servers_from_config,
    parse_mcp_server_spec,
)


class TestMCPServerConfig:
    def test_defaults(self):
        c = MCPServerConfig(name="t", transport="stdio")
        assert c.command is None
        assert c.args == []
        assert c.url is None
        assert c.env is None


class TestParseMcpServerSpec:
    def test_stdio_basic(self):
        c = parse_mcp_server_spec("stdio:weather:python")
        assert c.transport == "stdio"
        assert c.name == "weather"
        assert c.command == "python"
        assert c.args == []

    def test_stdio_with_args(self):
        c = parse_mcp_server_spec("stdio:weather:python:-m,server.py")
        assert c.command == "python"
        assert c.args == ["-m", "server.py"]

    def test_sse(self):
        c = parse_mcp_server_spec("sse:remote:http://localhost:8080/sse")
        assert c.transport == "sse"
        assert c.name == "remote"
        assert c.url == "http://localhost:8080/sse"

    def test_sse_url_with_colon(self):
        c = parse_mcp_server_spec("sse:remote:http://localhost:8080")
        assert c.url == "http://localhost:8080"

    def test_too_few_parts(self):
        with pytest.raises(ValueError):
            parse_mcp_server_spec("stdio:weather")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_mcp_server_spec("")

    def test_unsupported_transport(self):
        with pytest.raises(ValueError):
            parse_mcp_server_spec("ftp:name:host")

    def test_stdio_args_with_colon_in_command(self):
        c = parse_mcp_server_spec("stdio:db:python:script.py,arg1")
        assert c.command == "python"
        assert c.args == ["script.py", "arg1"]


class TestLoadMcpServersFromConfig:
    def test_valid_stdio(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "weather": {
                            "command": "python",
                            "args": ["server.py"],
                            "transport": "stdio",
                        }
                    }
                }
            )
        )
        configs = load_mcp_servers_from_config(str(f))
        assert len(configs) == 1
        assert configs[0].name == "weather"
        assert configs[0].transport == "stdio"
        assert configs[0].command == "python"

    def test_valid_sse(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "remote": {"url": "http://example.com/sse", "transport": "sse"}
                    }
                }
            )
        )
        configs = load_mcp_servers_from_config(str(f))
        assert len(configs) == 1
        assert configs[0].url == "http://example.com/sse"

    def test_multiple_servers(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "weather": {"command": "python", "transport": "stdio"},
                        "remote": {"url": "http://x.com/sse", "transport": "sse"},
                    }
                }
            )
        )
        configs = load_mcp_servers_from_config(str(f))
        assert len(configs) == 2

    def test_empty_mcpServers(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({"mcpServers": {}}))
        assert load_mcp_servers_from_config(str(f)) == []

    def test_missing_mcpServers_key(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({}))
        assert load_mcp_servers_from_config(str(f)) == []

    def test_mcpServers_not_dict(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({"mcpServers": []}))
        with pytest.raises(ValueError):
            load_mcp_servers_from_config(str(f))

    def test_stdio_missing_command(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({"mcpServers": {"x": {"transport": "stdio"}}}))
        with pytest.raises(ValueError):
            load_mcp_servers_from_config(str(f))

    def test_sse_missing_url(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({"mcpServers": {"x": {"transport": "sse"}}}))
        with pytest.raises(ValueError):
            load_mcp_servers_from_config(str(f))

    def test_default_transport_is_stdio(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps({"mcpServers": {"x": {"command": "python"}}}))
        configs = load_mcp_servers_from_config(str(f))
        assert configs[0].transport == "stdio"

    def test_invalid_transport(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(
            json.dumps({"mcpServers": {"x": {"transport": "ftp", "command": "y"}}})
        )
        with pytest.raises(ValueError):
            load_mcp_servers_from_config(str(f))

    def test_with_env(self, tmp_path):
        f = tmp_path / "mcp.json"
        f.write_text(
            json.dumps(
                {"mcpServers": {"x": {"command": "python", "env": {"DEBUG": "true"}}}}
            )
        )
        configs = load_mcp_servers_from_config(str(f))
        assert configs[0].env == {"DEBUG": "true"}


class TestMCPClientState:
    def test_list_tools_not_connected_raises(self):
        c = MCPClient(MCPServerConfig(name="t", transport="stdio", command="x"))
        with pytest.raises(RuntimeError, match="未连接"):
            c.list_tools_sync()

    def test_call_tool_not_connected_raises(self):
        c = MCPClient(MCPServerConfig(name="t", transport="stdio", command="x"))
        with pytest.raises(RuntimeError, match="未连接"):
            c.call_tool_sync("tool", {})

    def test_disconnect_never_connected(self):
        c = MCPClient(MCPServerConfig(name="t", transport="stdio", command="x"))
        c.disconnect()

    def test_build_target_stdio(self):
        c = MCPClient(
            MCPServerConfig(name="t", transport="stdio", command="python", args=["a"])
        )
        target = c._build_target()
        assert target is not None
        assert target.command == "python"
        assert target.args == ["a"]
        assert target.env["NPM_CONFIG_UPDATE_NOTIFIER"] == "false"
        assert target.env["NPM_CONFIG_FUND"] == "false"
        assert target.env["NPM_CONFIG_AUDIT"] == "false"

    def test_build_target_stdio_user_env_overrides_default(self):
        c = MCPClient(
            MCPServerConfig(
                name="t",
                transport="stdio",
                command="npx",
                args=["a"],
                env={"NPM_CONFIG_FUND": "true", "FOO": "1"},
            )
        )
        target = c._build_target()
        assert target.env["NPM_CONFIG_UPDATE_NOTIFIER"] == "false"
        assert target.env["NPM_CONFIG_FUND"] == "true"
        assert target.env["FOO"] == "1"

    def test_build_target_sse(self):
        c = MCPClient(MCPServerConfig(name="t", transport="sse", url="http://x.com"))
        target = c._build_target()
        assert target == "http://x.com"

    def test_build_target_unsupported(self):
        c = MCPClient(MCPServerConfig(name="t", transport="ftp"))
        with pytest.raises(ValueError):
            c._build_target()
