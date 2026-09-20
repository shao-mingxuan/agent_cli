.PHONY: chat run install

# MCP server 配置文件（替代长命令行参数）
MCP_CONFIG := ./mcp.json

chat:
	venv/bin/agent chat --mcp-config "$(MCP_CONFIG)"

run:
	venv/bin/agent run --mcp-config "$(MCP_CONFIG)" -p "$(p)"

install:
	venv/bin/python -m pip install -e .
