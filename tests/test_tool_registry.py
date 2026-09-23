"""L3 ToolRegistry 测试。"""

from agent_cli.tools.registry import ToolInfo, ToolRegistry, create_default_registry


class TestToolInfo:
    def test_defaults(self):
        info = ToolInfo(tool=object(), name="t")
        assert info.source == "builtin"
        assert info.category == ""

    def test_custom_fields(self):
        tool = object()
        info = ToolInfo(tool=tool, name="t", source="mcp", category="weather")
        assert info.source == "mcp"
        assert info.category == "weather"
        assert info.tool is tool


class TestToolRegistry:
    def test_register_single_tool(self):
        reg = ToolRegistry()
        tool = type("T", (), {"name": "my_tool"})()
        reg.register(tool)
        assert len(reg.get_all()) == 1
        assert reg.get_by_name("my_tool") is tool

    def test_register_tool_with_dunder_name(self):
        reg = ToolRegistry()
        reg.register(lambda: None)
        entries = reg.get_all_info()
        assert len(entries) == 1

    def test_register_many(self):
        reg = ToolRegistry()
        t1 = type("T", (), {"name": "t1"})()
        t2 = type("T", (), {"name": "t2"})()
        reg.register_many([t1, t2])
        assert len(reg.get_all()) == 2

    def test_get_all_returns_tools(self):
        reg = ToolRegistry()
        t1 = type("T", (), {"name": "t1"})()
        reg.register(t1)
        result = reg.get_all()
        assert result == [t1]

    def test_get_all_info_returns_toolinfo(self):
        reg = ToolRegistry()
        reg.register(type("T", (), {"name": "t1"})())
        result = reg.get_all_info()
        assert len(result) == 1
        assert isinstance(result[0], ToolInfo)

    def test_get_all_info_returns_copy(self):
        reg = ToolRegistry()
        reg.register(type("T", (), {"name": "t1"})())
        infos = reg.get_all_info()
        infos.clear()
        assert len(reg.get_all_info()) == 1

    def test_get_by_name_found(self):
        reg = ToolRegistry()
        t = type("T", (), {"name": "t1"})()
        reg.register(t)
        assert reg.get_by_name("t1") is t

    def test_get_by_name_not_found(self):
        reg = ToolRegistry()
        assert reg.get_by_name("nope") is None

    def test_get_info_by_name_found(self):
        reg = ToolRegistry()
        reg.register(type("T", (), {"name": "t1"})())
        info = reg.get_info_by_name("t1")
        assert info is not None
        assert info.name == "t1"

    def test_get_info_by_name_not_found(self):
        reg = ToolRegistry()
        assert reg.get_info_by_name("nope") is None

    def test_clear_empties_registry(self):
        reg = ToolRegistry()
        reg.register(type("T", (), {"name": "t1"})())
        reg.register(type("T", (), {"name": "t2"})())
        reg.register(type("T", (), {"name": "t3"})())
        reg.clear()
        assert reg.get_all() == []

    def test_register_preserves_source_and_category(self):
        reg = ToolRegistry()
        reg.register(type("T", (), {"name": "t"})(), source="mcp", category="weather")
        info = reg.get_info_by_name("t")
        assert info.source == "mcp"
        assert info.category == "weather"


class TestCreateDefaultRegistry:
    def test_has_5_tools(self):
        reg = create_default_registry()
        assert len(reg.get_all()) == 5

    def test_tool_names(self):
        reg = create_default_registry()
        names = [info.name for info in reg.get_all_info()]
        assert "calculate" in names
        assert "detect_pii" in names
        assert "detect_violation" in names
        assert "scan_file" in names
        assert "validate_input" in names

    def test_sources_all_builtin(self):
        reg = create_default_registry()
        for info in reg.get_all_info():
            assert info.source == "builtin"
