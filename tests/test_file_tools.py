"""L3 文件读写工具测试。"""

from agent_cli.tools.builtin.file_io import (
    edit_file,
    list_directory,
    read_file,
    search_files,
    write_file,
)


class TestReadFile:
    def test_missing_file(self, tmp_path):
        assert "Error" in read_file.invoke({"path": str(tmp_path / "nope.txt")})

    def test_directory(self, tmp_path):
        assert "Error" in read_file.invoke({"path": str(tmp_path)})

    def test_read_all(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("line1\nline2\nline3\n")
        out = read_file.invoke({"path": str(p)})
        assert "line1" in out
        assert "3 lines" in out

    def test_read_offset_limit(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("\n".join(f"line{i}" for i in range(10)))
        out = read_file.invoke({"path": str(p), "offset": 2, "limit": 2})
        assert "line1" in out and "line2" in out
        assert "line0" not in out


class TestWriteFile:
    def test_create(self, tmp_path):
        p = tmp_path / "b.txt"
        res = write_file.invoke({"path": str(p), "content": "hello"})
        assert "已写入" in res
        assert p.read_text() == "hello"

    def test_create_nested(self, tmp_path):
        p = tmp_path / "x" / "y" / "c.txt"
        res = write_file.invoke({"path": str(p), "content": "nested"})
        assert "已写入" in res
        assert p.read_text() == "nested"

    def test_overwrite(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("old")
        write_file.invoke({"path": str(p), "content": "new"})
        assert p.read_text() == "new"


class TestEditFile:
    def test_missing_file(self, tmp_path):
        res = edit_file.invoke(
            {"path": str(tmp_path / "nope.txt"), "old_string": "a", "new_string": "b"}
        )
        assert "Error" in res

    def test_replace_once(self, tmp_path):
        p = tmp_path / "e.txt"
        p.write_text("hello world")
        res = edit_file.invoke(
            {"path": str(p), "old_string": "world", "new_string": "kitty"}
        )
        assert "已修改" in res
        assert p.read_text() == "hello kitty"

    def test_not_found(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("abc")
        res = edit_file.invoke(
            {"path": str(p), "old_string": "xyz", "new_string": "uvw"}
        )
        assert "Error" in res

    def test_multiple_match_requires_context(self, tmp_path):
        p = tmp_path / "g.txt"
        p.write_text("a\na\na")
        res = edit_file.invoke(
            {"path": str(p), "old_string": "a", "new_string": "b"}
        )
        assert "Error" in res

    def test_replace_all(self, tmp_path):
        p = tmp_path / "h.txt"
        p.write_text("a\na\na")
        res = edit_file.invoke(
            {"path": str(p), "old_string": "a", "new_string": "b", "replace_all": True}
        )
        assert p.read_text() == "b\nb\nb"


class TestListDirectory:
    def test_missing(self, tmp_path):
        assert "Error" in list_directory.invoke({"path": str(tmp_path / "gone")})

    def test_lists_entries(self, tmp_path):
        (tmp_path / "f1.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        out = list_directory.invoke({"path": str(tmp_path)})
        assert "f1.txt" in out
        assert "sub" in out
        assert "[目录]" in out


class TestSearchFiles:
    def test_find_py(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        out = search_files.invoke({"directory": str(tmp_path), "pattern": "*.py"})
        assert "a.py" in out
        assert "b.py" in out
        assert "c.txt" not in out

    def test_no_match(self, tmp_path):
        out = search_files.invoke({"directory": str(tmp_path), "pattern": "*.rar"})
        assert "未找到" in out

    def test_missing_dir(self, tmp_path):
        out = search_files.invoke(
            {"directory": str(tmp_path / "gone"), "pattern": "*.py"}
        )
        assert "Error" in out
