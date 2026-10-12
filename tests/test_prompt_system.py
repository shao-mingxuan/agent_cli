"""L2 prompt 系统测试：renderer / loader / manager。"""

import pytest

from agent_cli.prompts.loader import load_prompt_dir, load_prompt_file, parse_frontmatter
from agent_cli.prompts.manager import PromptManager, create_prompt_manager
from agent_cli.prompts.renderer import render_prompt, render_template


class TestRenderer:
    def test_render_basic(self):
        assert render_template("你好 {{name}}", {"name": "peter"}) == "你好 peter"

    def test_missing_var_kept(self):
        assert render_template("{{a}}-{{b}}", {"a": "1"}) == "1-{{b}}"

    def test_value_none_kept(self):
        assert render_template("{{a}}", {"a": None}) == "{{a}}"

    def test_non_string_values(self):
        assert render_template("count={{n}}", {"n": 3}) == "count=3"

    def test_render_prompt_kwargs(self):
        assert render_prompt("hi {{who}}", who="world") == "hi world"

    def test_empty_template(self):
        assert render_template("") == ""


class TestLoader:
    def test_parse_frontmatter(self):
        text = "---\nname: my_prompt\ndescription: desc\n---\nbody text"
        meta, body = parse_frontmatter(text)
        assert meta == {"name": "my_prompt", "description": "desc"}
        assert body == "body text"

    def test_parse_no_frontmatter(self):
        assert parse_frontmatter("no meta here") == ({}, "no meta here")

    def test_parse_quoted_values(self):
        meta, _ = parse_frontmatter('---\ntitle: "my title"\n---\nbody')
        assert meta["title"] == "my title"

    def test_load_prompt_file_missing(self, tmp_path):
        assert load_prompt_file(str(tmp_path / "nope.md")) is None

    def test_load_prompt_file_with_meta(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("---\nname: custom\ndescription: d\n---\n这是一个 prompt")
        data = load_prompt_file(str(p))
        assert data["name"] == "custom"
        assert data["description"] == "d"
        assert data["system_prompt"] == "这是一个 prompt"
        assert data["source"] == str(p)

    def test_load_prompt_file_uses_filename(self, tmp_path):
        p = tmp_path / "hello.md"
        p.write_text("body only")
        data = load_prompt_file(str(p))
        assert data["name"] == "hello"

    def test_load_prompt_dir(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nname: A\n---\n内容A")
        (tmp_path / "b.txt").write_text("内容B")
        (tmp_path / "_skip.md").write_text("skip")
        (tmp_path / "readme.txt").write_text("nope")  # frontmatter 文件

        prompts = load_prompt_dir(str(tmp_path))
        names = {p["name"] for p in prompts}
        assert {"A", "b"} <= names
        assert any(p["system_prompt"] == "内容A" for p in prompts)

    def test_load_prompt_dir_nonexistent(self):
        assert load_prompt_dir("/nonexistent") == []

    def test_empty_body_skipped(self, tmp_path):
        (tmp_path / "empty.txt").write_text("---\nname: x\n---\n   ")
        assert load_prompt_dir(str(tmp_path)) == []


class TestManager:
    def test_init_default(self):
        pm = create_prompt_manager()
        assert pm.current == "default"
        assert "coder" in pm.list()

    def test_get_current(self):
        pm = PromptManager({"a": "prompt a"})
        assert pm.get() == "prompt a"

    def test_get_named(self):
        pm = PromptManager({"a": "1", "b": "2"})
        assert pm.get("b") == "2"

    def test_register_and_switch(self):
        pm = PromptManager({"a": "1"})
        pm.register("b", "2")
        assert pm.set_current("b") == "2"
        assert pm.current == "b"

    def test_unknown_current_raises(self):
        with pytest.raises(KeyError):
            PromptManager({"a": "1"}, current="b")

    def test_unknown_switch_raises(self):
        pm = PromptManager({"a": "1"})
        with pytest.raises(KeyError):
            pm.set_current("nope")

    def test_rollback(self):
        pm = PromptManager({"a": "1", "b": "2"}, current="a")
        pm.set_current("b")
        assert pm.rollback() == "a"
        assert pm.rollback() is None

    def test_rollback_with_no_history(self):
        pm = PromptManager({"a": "1"})
        assert pm.rollback() is None

    def test_empty_manager(self):
        pm = PromptManager()
        assert pm.current is None
        assert pm.get() == ""
