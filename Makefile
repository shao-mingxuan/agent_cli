.PHONY: chat run install

WEATHER_MCP := stdio:weather:venv/bin/python:mcp_servers/weather_server.py
FS_MCP := stdio:fs:npx:-y,@modelcontextprotocol/server-filesystem,/tmp

chat:
	venv/bin/agent chat --mcp-server "$(WEATHER_MCP)" --mcp-server "$(FS_MCP)"

run:
	venv/bin/agent run --mcp-server "$(WEATHER_MCP)" --mcp-server "$(FS_MCP)" -p "$(p)"

install:
	venv/bin/python -m pip install -e .
