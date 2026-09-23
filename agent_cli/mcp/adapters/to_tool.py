"""L3 MCP 适配器 - 将 MCP Tool 转为 LangChain BaseTool。"""

from typing import Any, Type, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from ..client import MCPClient

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_type_to_python(
    json_type: str, schema: dict | None = None, model_name: str = "NestedModel"
) -> type:
    """将单个 JSON Schema type 映射为 Python 类型。"""
    base = _JSON_TYPE_MAP.get(json_type, str)

    if json_type == "object" and schema is not None:
        nested = _schema_to_pydantic_model(schema, model_name)
        if nested is not None:
            return nested

    if json_type == "array" and schema is not None:
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            item_type = _schema_node_to_python(items_schema, f"{model_name}Item")
            return list[item_type]

    return base


def _schema_node_to_python(node: dict, model_name: str = "NestedModel") -> type:
    """将一个 JSON Schema 节点（可能是 $ref / enum / oneOf / allOf / 嵌套）映射为 Python 类型。

    返回的 type 可以直接用于 pydantic Field 注解。
    """
    if not isinstance(node, dict):
        return str

    # enum → Literal
    enum_values = node.get("enum")
    if enum_values is not None:
        from typing import Literal

        return Literal[tuple(enum_values)]

    # anyOf / oneOf → Union
    for combiner_key in ("anyOf", "oneOf"):
        sub_schemas = node.get(combiner_key)
        if isinstance(sub_schemas, list) and sub_schemas:
            sub_types = tuple(
                _schema_node_to_python(s, model_name) for s in sub_schemas
            )
            if len(sub_types) == 1:
                return sub_types[0]
            return Union[sub_types]

    # allOf → 合并后递归（取第一个子 schema 的 properties 合并）
    all_of = node.get("allOf")
    if isinstance(all_of, list) and all_of:
        merged: dict = {}
        for sub in all_of:
            if isinstance(sub, dict):
                merged.update(sub.get("properties", {}))
                if "required" not in merged and "required" in sub:
                    merged["required"] = sub["required"]
        if merged:
            node = {**node, "properties": {**node.get("properties", {}), **merged}}

    # 有 properties → 嵌套 pydantic model
    if "properties" in node or node.get("type") == "object":
        nested = _schema_to_pydantic_model(node, model_name)
        if nested is not None:
            return nested

    # 普通类型
    json_type = node.get("type")
    if json_type is None:
        if "enum" in node:
            from typing import Literal

            return Literal[tuple(node["enum"])]
        return str

    if isinstance(json_type, list):
        types = [
            _json_type_to_python(t, node, model_name) for t in json_type if t != "null"
        ]
        has_null = "null" in json_type
        if not types:
            return type(None)
        if len(types) == 1:
            return types[0] | None if has_null else types[0]
        result = Union[tuple(types)]
        return result | None if has_null else result

    return _json_type_to_python(json_type, node, model_name)


def _schema_to_pydantic_model(
    schema: dict, model_name: str = "MCPToolArgs"
) -> Type[BaseModel] | None:
    """将一个 JSON Schema（object 类型）转为 Pydantic model。

    返回 None 表示无法构建（例如没有 properties 的空对象）。
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return None

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            prop_schema = {"type": "string"}

        description = prop_schema.get("description", "")
        py_type = _schema_node_to_python(prop_schema, f"{model_name}_{prop_name}")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (
                py_type | None,
                Field(default=default, description=description),
            )

    return create_model(model_name, **fields)


def _json_schema_to_pydantic(
    schema: dict, model_name: str = "MCPToolArgs"
) -> Type[BaseModel]:
    """将 MCP 工具的 inputSchema 转为 Pydantic model。

    支持：
    - 基础类型: string/integer/number/boolean/array/object/null
    - enum → Literal
    - anyOf / oneOf → Union
    - allOf → 合并 properties
    - 嵌套 object → 递归生成子 model
    - array.items → list[item_type]
    - type 数组 (如 ["string", "null"]) → Optional
    - $ref → 简化处理为 str（MCP 工具一般不用 $ref）
    """
    model = _schema_to_pydantic_model(schema, model_name)
    if model is not None:
        return model

    return create_model(model_name)


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
