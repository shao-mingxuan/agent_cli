"""L3 MCP 适配器 - 将 MCP Tool 转为 LangChain BaseTool。"""
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from ..client import MCPClient


def _json_schema_to_pydantic(schema: dict, model_name: str = "MCPToolArgs") -> Type[BaseModel]:
    """将 JSON Schema 转为 Pydantic model。"""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, tuple] = {}
    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type = _json_type_to_python(json_type)
        description = prop_schema.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, ...)
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (py_type | None, default)

    if not fields:
        return create_model(model_name)

    return create_model(model_name, **fields)


def _json_type_to_python(json_type: str) -> type:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


class MCPToolAdapter(BaseTool):
    """将 MCP Tool 适配为 LangChain BaseTool。"""

    _client: MCPClient
    _original_name: str

    def __init__(self, mcp_tool: Any, client: MCPClient, server_name: str):
        original_name = mcp_tool.name
        prefixed_name = f"{server_name}_{original_name}"
        description = mcp_tool.description or f"MCP tool: {original_name}"

        args_schema = _json_schema_to_pydantic(
            mcp_tool.input_schema,
            model_name=f"{server_name}_{original_name}_args",
        )

        super().__init__(
            name=prefixed_name,
            description=description,
            args_schema=args_schema,
        )
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_original_name", original_name)

    def _run(self, **kwargs: Any) -> str:
        return self._client.call_tool_sync(self._original_name, kwargs)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)
