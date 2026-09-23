"""L3 MCP 注册中心 - 管理多个 MCP server 连接。"""

from ..tools.registry import ToolInfo
from .adapters.to_tool import MCPToolAdapter
from .client import MCPClient, MCPServerConfig


class MCPRegistry:
    """管理多个 MCP server 连接，聚合所有工具。"""

    def __init__(self):
        self._clients: list[MCPClient] = []
        self._tool_infos: list[ToolInfo] = []

    def add_server(self, config: MCPServerConfig) -> int:
        """连接 server 并注册其所有工具，返回工具数量。

        连接失败或工具列表获取失败时返回 0，不中断后续 server 加载。
        """
        client = MCPClient(config)
        try:
            client.connect()
        except Exception as e:
            from rich.console import Console

            Console().print(
                f"[dim][MCP] 连接 '{config.name}' 失败 ({config.transport}): {e}[/dim]"
            )
            return 0

        self._clients.append(client)

        try:
            mcp_tools = client.list_tools_sync()
        except Exception as e:
            from rich.console import Console

            Console().print(f"[dim][MCP] '{config.name}' 工具列表获取失败: {e}[/dim]")
            return 0

        count = 0
        for mcp_tool in mcp_tools:
            adapter = MCPToolAdapter(
                mcp_tool=mcp_tool,
                client=client,
                server_name=config.name,
            )
            self._tool_infos.append(
                ToolInfo(
                    tool=adapter,
                    name=adapter.name,
                    source="mcp",
                    category=config.name,
                )
            )
            count += 1
        return count

    def get_tool_infos(self) -> list[ToolInfo]:
        return list(self._tool_infos)

    def get_tools(self) -> list:
        return [info.tool for info in self._tool_infos]

    def disconnect_all(self) -> None:
        for client in self._clients:
            client.disconnect()
        self._clients.clear()
        self._tool_infos.clear()
