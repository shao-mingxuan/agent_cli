"""L3 MCP 适配器测试 - JSON Schema→Pydantic 转换 + MCPToolAdapter。"""
import asyncio
from types import SimpleNamespace
from typing import get_args, get_origin, Union, Literal

import pytest
from pydantic import BaseModel

from agent_cli.mcp.adapters.to_tool import (
    _json_type_to_python,
    _schema_node_to_python,
    _schema_to_pydantic_model,
    _json_schema_to_pydantic,
    MCPToolAdapter,
)


class TestJsonTypeToPython:
    def test_string(self):
        assert _json_type_to_python("string") is str

    def test_integer(self):
        assert _json_type_to_python("integer") is int

    def test_number(self):
        assert _json_type_to_python("number") is float

    def test_boolean(self):
        assert _json_type_to_python("boolean") is bool

    def test_array_no_schema(self):
        assert _json_type_to_python("array") is list

    def test_object_no_schema(self):
        assert _json_type_to_python("object") is dict

    def test_null(self):
        assert _json_type_to_python("null") is type(None)

    def test_unknown_type_falls_back_to_str(self):
        assert _json_type_to_python("foobar") is str

    def test_array_with_items(self):
        result = _json_type_to_python("array", {"items": {"type": "string"}}, "Test")
        assert result == list[str]


class TestSchemaNodeToPython:
    def test_enum(self):
        result = _schema_node_to_python({"enum": ["a", "b"]})
        assert get_origin(result) is Literal
        assert get_args(result) == ("a", "b")

    def test_anyof_two_types(self):
        result = _schema_node_to_python({
            "anyOf": [{"type": "string"}, {"type": "integer"}]
        })
        assert get_origin(result) is Union
        assert str in get_args(result)
        assert int in get_args(result)

    def test_oneof_two_types(self):
        result = _schema_node_to_python({
            "oneOf": [{"type": "string"}, {"type": "null"}]
        })
        assert get_origin(result) is Union

    def test_anyof_single_type(self):
        result = _schema_node_to_python({"anyOf": [{"type": "string"}]})
        assert result is str

    def test_allof_merges_properties(self):
        result = _schema_node_to_python({
            "allOf": [{"properties": {"a": {"type": "string"}}}]
        })
        assert result is not None

    def test_nested_object(self):
        result = _schema_node_to_python({
            "type": "object",
            "properties": {"x": {"type": "string"}}
        })
        assert isinstance(result, type)
        assert issubclass(result, BaseModel)

    def test_array_with_items(self):
        result = _schema_node_to_python({
            "type": "array",
            "items": {"type": "string"}
        })
        assert result == list[str]

    def test_type_array_optional(self):
        result = _schema_node_to_python({"type": ["string", "null"]})
        args = get_args(result)
        assert str in args
        assert type(None) in args

    def test_type_array_no_null(self):
        result = _schema_node_to_python({"type": ["string", "integer"]})
        assert get_origin(result) is Union

    def test_no_type_no_enum_falls_back_to_str(self):
        result = _schema_node_to_python({})
        assert result is str

    def test_non_dict_returns_str(self):
        result = _schema_node_to_python("not a dict")
        assert result is str

    def test_enum_without_type(self):
        result = _schema_node_to_python({"enum": [1, 2, 3]})
        assert get_origin(result) is Literal


class TestSchemaToPydanticModel:
    def test_with_required_field(self):
        model = _schema_to_pydantic_model({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        })
        assert model is not None
        assert issubclass(model, BaseModel)
        instance = model(name="test")
        assert instance.name == "test"

    def test_with_optional_field(self):
        model = _schema_to_pydantic_model({
            "type": "object",
            "properties": {"age": {"type": "integer"}}
        })
        assert model is not None
        instance = model()
        assert instance.age is None

    def test_no_properties_returns_none(self):
        result = _schema_to_pydantic_model({"type": "object"})
        assert result is None

    def test_empty_properties_returns_none(self):
        result = _schema_to_pydantic_model({"properties": {}})
        assert result is None


class TestJsonSchemaToPydantic:
    def test_with_properties(self):
        model = _json_schema_to_pydantic({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        })
        assert issubclass(model, BaseModel)
        instance = model(q="hello")
        assert instance.q == "hello"

    def test_empty_schema_returns_bare_model(self):
        model = _json_schema_to_pydantic({})
        assert issubclass(model, BaseModel)
        instance = model()
        assert isinstance(instance, BaseModel)


class TestMCPToolAdapter:
    def _make_mock_tool(self, name="get_weather", description="Get weather", schema=None):
        return SimpleNamespace(
            name=name,
            description=description,
            input_schema=schema or {"type": "object", "properties": {}},
        )

    def _make_mock_client(self, return_value="ok"):
        client = SimpleNamespace()
        client.call_tool_sync = lambda name, args: return_value
        return client

    def test_name_prefixed(self):
        tool = self._make_mock_tool()
        adapter = MCPToolAdapter(tool, self._make_mock_client(), "weather")
        assert adapter.name == "weather_get_weather"

    def test_description_from_tool(self):
        tool = self._make_mock_tool(description="Get weather info")
        adapter = MCPToolAdapter(tool, self._make_mock_client(), "weather")
        assert adapter.description == "Get weather info"

    def test_description_fallback(self):
        tool = self._make_mock_tool(description=None)
        adapter = MCPToolAdapter(tool, self._make_mock_client(), "weather")
        assert "get_weather" in adapter.description

    def test_run_delegates_to_client(self):
        tool = self._make_mock_tool()
        client = self._make_mock_client("sunny")
        adapter = MCPToolAdapter(tool, client, "weather")
        result = adapter._run(city="test")
        assert result == "sunny"

    def test_run_uses_original_name(self):
        tool = self._make_mock_tool()
        captured_name = []

        def capture(name, args):
            captured_name.append(name)
            return "ok"

        client = SimpleNamespace(call_tool_sync=capture)
        adapter = MCPToolAdapter(tool, client, "weather")
        adapter._run(city="x")
        assert captured_name == ["get_weather"]

    def test_arun_returns_same_as_run(self):
        tool = self._make_mock_tool()
        adapter = MCPToolAdapter(tool, self._make_mock_client("result"), "weather")
        result = asyncio.run(adapter._arun(city="x"))
        assert result == "result"
