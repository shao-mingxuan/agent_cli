"""L3 MCPRegistry 测试。"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent_cli.mcp.client import MCPServerConfig
from agent_cli.mcp.registry import MCPRegistry


def _make_mock_mcp_tool(name="get_weather", description="Get weather"):
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
    )


class TestEmptyRegistry:
    def test_get_tool_infos(self):
        assert MCPRegistry().get_tool_infos() == []

    def test_get_tools(self):
        assert MCPRegistry().get_tools() == []

    def test_disconnect_all(self):
        reg = MCPRegistry()
        reg.disconnect_all()
        assert reg.get_tool_infos() == []

    def test_disconnect_all_calls_each_client(self):
        reg = MCPRegistry()
        client1 = MagicMock()
        client2 = MagicMock()
        reg._clients = [client1, client2]
        reg.disconnect_all()
        client1.disconnect.assert_called_once()
        client2.disconnect.assert_called_once()
        assert reg.get_tool_infos() == []

    def test_disconnect_all_parallel(self):
        import threading
        import time
        from unittest.mock import MagicMock

        started = []
        lock = threading.Lock()

        def slow_disconnect():
            with lock:
                started.append(True)
            time.sleep(0.4)

        reg = MCPRegistry()
        client1 = MagicMock()
        client2 = MagicMock()
        client1.disconnect.side_effect = slow_disconnect
        client2.disconnect.side_effect = slow_disconnect
        reg._clients = [client1, client2]

        t0 = time.monotonic()
        reg.disconnect_all()
        elapsed = time.monotonic() - t0

        assert client1.disconnect.call_count == 1
        assert client2.disconnect.call_count == 1
        assert len(started) == 2
        # 串行需要 0.8s，并行应在单个耗时附近完成
        assert elapsed < 0.79, f"disconnect_all 未并行执行: {elapsed:.2f}s"


class TestAddServer:
    def test_success(self):
        reg = MCPRegistry()
        mock_client = MagicMock()
        mock_client.connect = MagicMock()
        mock_client.list_tools_sync = MagicMock(
            return_value=[_make_mock_mcp_tool("t1"), _make_mock_mcp_tool("t2")]
        )

        with patch("agent_cli.mcp.registry.MCPClient", return_value=mock_client):
            count = reg.add_server(
                MCPServerConfig(name="srv", transport="stdio", command="x")
            )

        assert count == 2
        assert len(reg.get_tool_infos()) == 2
        assert reg.get_tool_infos()[0].source == "mcp"
        assert reg.get_tool_infos()[0].category == "srv"

    def test_connect_failure_returns_zero(self):
        reg = MCPRegistry()
        mock_client = MagicMock()
        mock_client.connect = MagicMock(side_effect=ConnectionError("failed"))

        with patch("agent_cli.mcp.registry.MCPClient", return_value=mock_client):
            count = reg.add_server(
                MCPServerConfig(name="srv", transport="stdio", command="x")
            )

        assert count == 0
        assert reg.get_tool_infos() == []

    def test_list_tools_failure_returns_zero(self):
        reg = MCPRegistry()
        mock_client = MagicMock()
        mock_client.connect = MagicMock()
        mock_client.list_tools_sync = MagicMock(side_effect=RuntimeError("failed"))

        with patch("agent_cli.mcp.registry.MCPClient", return_value=mock_client):
            count = reg.add_server(
                MCPServerConfig(name="srv", transport="stdio", command="x")
            )

        assert count == 0
        assert reg.get_tool_infos() == []
