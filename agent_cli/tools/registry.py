"""L3 工具 - 工具注册中心。"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """工具的元数据。"""

    tool: Any
    name: str
    source: str = "builtin"
    category: str = ""


class ToolRegistry:
    """工具注册中心，管理所有可用工具。"""

    def __init__(self):
        self._entries: list[ToolInfo] = []

    def register(
        self, tool: Any, source: str = "builtin", category: str = ""
    ) -> None:
        name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        self._entries.append(ToolInfo(tool, name, source, category))

    def register_many(
        self, tools: list[Any], source: str = "builtin", category: str = ""
    ) -> None:
        for t in tools:
            self.register(t, source=source, category=category)

    def get_all(self) -> list[Any]:
        return [e.tool for e in self._entries]

    def get_all_info(self) -> list[ToolInfo]:
        return list(self._entries)

    def get_by_name(self, name: str) -> Any | None:
        for e in self._entries:
            if e.name == name:
                return e.tool
        return None

    def get_info_by_name(self, name: str) -> ToolInfo | None:
        for e in self._entries:
            if e.name == name:
                return e
        return None

    def clear(self) -> None:
        self._entries.clear()


def create_default_registry() -> ToolRegistry:
    """创建带有默认内置工具的注册中心。"""
    from .builtin.weather import get_weather
    from .builtin.calculate import calculate
    from .builtin.safety import detect_pii, detect_violation, scan_file, validate_input

    registry = ToolRegistry()
    registry.register(get_weather, source="builtin", category="weather")
    registry.register(calculate, source="builtin", category="calculate")
    registry.register(detect_pii, source="builtin", category="safety")
    registry.register(detect_violation, source="builtin", category="safety")
    registry.register(scan_file, source="builtin", category="safety")
    registry.register(validate_input, source="builtin", category="safety")
    return registry
