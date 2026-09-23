"""L3 插件加载器测试。"""

import sys

from agent_cli.tools.loader import load_plugins

PLUGIN_CODE = '''
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """A test tool."""
    return f"result: {query}"
'''

BAD_PLUGIN_CODE = """
import nonexistent_module_xyz
"""

NON_TOOL_CODE = """
def regular_function():
    return "not a tool"

class SomeClass:
    pass
"""


class TestLoadPlugins:
    def test_nonexistent_dir(self):
        assert load_plugins("/nonexistent/path") == []

    def test_empty_dir(self, tmp_path):
        assert load_plugins(str(tmp_path)) == []

    def test_valid_tool(self, tmp_path):
        (tmp_path / "mytool.py").write_text(PLUGIN_CODE)
        infos = load_plugins(str(tmp_path))
        assert len(infos) == 1
        assert infos[0].name == "my_tool"
        assert infos[0].source == "plugin"
        assert infos[0].category == "mytool"

    def test_underscore_files_skipped(self, tmp_path):
        (tmp_path / "_skip.py").write_text(PLUGIN_CODE)
        assert load_plugins(str(tmp_path)) == []

    def test_non_py_files_skipped(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not python")
        assert load_plugins(str(tmp_path)) == []

    def test_multiple_tools(self, tmp_path):
        (tmp_path / "a.py").write_text(PLUGIN_CODE)
        (tmp_path / "b.py").write_text(PLUGIN_CODE.replace("my_tool", "other_tool"))
        infos = load_plugins(str(tmp_path))
        assert len(infos) == 2

    def test_import_error_continues(self, tmp_path):
        (tmp_path / "bad.py").write_text(BAD_PLUGIN_CODE)
        (tmp_path / "good.py").write_text(PLUGIN_CODE)
        infos = load_plugins(str(tmp_path))
        assert len(infos) == 1
        assert infos[0].name == "my_tool"

    def test_non_tool_attributes_not_collected(self, tmp_path):
        (tmp_path / "misc.py").write_text(NON_TOOL_CODE)
        assert load_plugins(str(tmp_path)) == []

    def test_multiple_tools_per_file(self, tmp_path):
        code = PLUGIN_CODE + "\n\n" + PLUGIN_CODE.replace("my_tool", "second_tool")
        (tmp_path / "multi.py").write_text(code)
        infos = load_plugins(str(tmp_path))
        assert len(infos) == 2

    def test_cleanup_sys_modules(self, tmp_path):
        (tmp_path / "cleanup_test.py").write_text(PLUGIN_CODE)
        load_plugins(str(tmp_path))
        keys_to_clean = [k for k in sys.modules if k.startswith("plugin_")]
        for k in keys_to_clean:
            del sys.modules[k]
