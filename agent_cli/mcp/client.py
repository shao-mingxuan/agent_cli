"""L3 MCP 客户端 - 连接单个 MCP server，async→sync 桥接。"""
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import Tool as MCPTool, TextContent


@dataclass
class MCPServerConfig:
    """单个 MCP server 配置。"""

    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] | None = None


class MCPClient:
    """连接单个 MCP server，提供同步 list_tools / call_tool 接口。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client | None = None
        self._connected = False

    def connect(self) -> None:
        """启动后台事件循环并连接 MCP server。"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
        future.result(timeout=30)

    async def _async_connect(self) -> None:
        target = self._build_target()
        self._client = Client(target)
        await self._client.__aenter__()
        self._connected = True

    def _build_target(self) -> Any:
        if self.config.transport == "stdio":
            return StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
        elif self.config.transport == "sse":
            return self.config.url
        else:
            raise ValueError(f"不支持的传输方式: {self.config.transport}")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def list_tools_sync(self) -> list[MCPTool]:
        """同步获取 server 上的工具列表。"""
        if not self._connected:
            raise RuntimeError("MCP client 未连接")
        future = asyncio.run_coroutine_threadsafe(self._async_list_tools(), self._loop)
        result = future.result(timeout=30)
        return result

    async def _async_list_tools(self) -> list[MCPTool]:
        result = await self._client.list_tools()
        return result.tools

    def call_tool_sync(self, name: str, arguments: dict) -> str:
        """同步调用工具并返回文本结果。"""
        if not self._connected:
            raise RuntimeError("MCP client 未连接")
        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(name, arguments), self._loop
        )
        result = future.result(timeout=60)
        return result

    async def _async_call_tool(self, name: str, arguments: dict) -> str:
        result = await self._client.call_tool(name, arguments)
        texts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                texts.append(item.text)
            else:
                texts.append(str(item))
        output = "\n".join(texts)
        if result.is_error:
            return f"[MCP 工具错误] {output}"
        return output

    def disconnect(self) -> None:
        """关闭连接并停止事件循环。"""
        if self._connected and self._client:
            future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._loop:
            self._loop.close()
        self._connected = False
        self._client = None
        self._loop = None
        self._thread = None

    async def _async_disconnect(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)


def parse_mcp_server_spec(spec: str) -> MCPServerConfig:
    """解析 CLI 参数为 MCPServerConfig。

    格式：
      stdio:name:command:arg1,arg2,arg3
      sse:name:url
    """
    parts = spec.split(":", 3)
    transport = parts[0]

    if transport == "stdio":
        if len(parts) < 3:
            raise ValueError(f"stdio 格式错误: {spec}，应为 stdio:name:command:arg1,arg2")
        name = parts[1]
        command = parts[2]
        args = parts[3].split(",") if len(parts) > 3 and parts[3] else []
        return MCPServerConfig(name=name, transport="stdio", command=command, args=args)

    elif transport == "sse":
        if len(parts) < 3:
            raise ValueError(f"sse 格式错误: {spec}，应为 sse:name:url")
        name = parts[1]
        url = parts[2]
        return MCPServerConfig(name=name, transport="sse", url=url)

    else:
        raise ValueError(f"不支持的传输方式: {transport}，支持 stdio 或 sse")
