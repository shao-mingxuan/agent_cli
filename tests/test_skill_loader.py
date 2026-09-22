"""L3 技能加载器测试。"""
import sys

from agent_cli.skills.loader import load_skill_modules, load_skill_registry
from agent_cli.skills.define_skill import Skill


SKILL_CODE = '''
from agent_cli.skills.define_skill import Skill

my_skill = Skill(name="my_skill", description="test", system_prompt="prompt")
'''

BAD_SKILL_CODE = '''
import nonexistent_module_xyz
'''

NON_SKILL_CODE = '''
class SomeClass:
    pass

def some_function():
    pass
'''


class TestLoadSkillModules:
    def test_nonexistent_dir(self):
        assert load_skill_modules("/nonexistent/path") == []

    def test_empty_dir(self, tmp_path):
        assert load_skill_modules(str(tmp_path)) == []

    def test_valid_skill(self, tmp_path):
        (tmp_path / "myskill.py").write_text(SKILL_CODE)
        skills = load_skill_modules(str(tmp_path))
        assert len(skills) == 1
        assert skills[0].name == "my_skill"

    def test_underscore_skipped(self, tmp_path):
        (tmp_path / "_skip.py").write_text(SKILL_CODE)
        assert load_skill_modules(str(tmp_path)) == []

    def test_non_py_skipped(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not python")
        assert load_skill_modules(str(tmp_path)) == []

    def test_import_error_continues(self, tmp_path):
        (tmp_path / "bad.py").write_text(BAD_SKILL_CODE)
        (tmp_path / "good.py").write_text(SKILL_CODE)
        skills = load_skill_modules(str(tmp_path))
        assert len(skills) == 1
        assert skills[0].name == "my_skill"

    def test_multiple_skills(self, tmp_path):
        (tmp_path / "a.py").write_text(SKILL_CODE)
        (tmp_path / "b.py").write_text(SKILL_CODE.replace("my_skill", "other_skill"))
        skills = load_skill_modules(str(tmp_path))
        assert len(skills) == 2

    def test_non_skill_attributes_not_collected(self, tmp_path):
        (tmp_path / "misc.py").write_text(NON_SKILL_CODE)
        assert load_skill_modules(str(tmp_path)) == []

    def test_multiple_per_file(self, tmp_path):
        code = SKILL_CODE + "\n\n" + SKILL_CODE.replace("my_skill", "second_skill")
        (tmp_path / "multi.py").write_text(code)
        skills = load_skill_modules(str(tmp_path))
        assert len(skills) == 2

    def test_cleanup_sys_modules(self, tmp_path):
        (tmp_path / "cleanup.py").write_text(SKILL_CODE)
        load_skill_modules(str(tmp_path))
        keys_to_clean = [k for k in sys.modules if k.startswith("user_skill_")]
        for k in keys_to_clean:
            del sys.modules[k]


class TestLoadSkillRegistry:
    def test_combines_default_and_external(self, tmp_path):
        (tmp_path / "custom.py").write_text(SKILL_CODE)
        reg = load_skill_registry(str(tmp_path))
        assert "research" in reg.list_names()
        assert "summarize_doc" in reg.list_names()
        assert "my_skill" in reg.list_names()

    def test_only_default(self, tmp_path):
        reg = load_skill_registry(str(tmp_path))
        assert len(reg.list_skills()) == 2
        assert "research" in reg.list_names()
        assert "summarize_doc" in reg.list_names()

    def test_cleanup_sys_modules(self, tmp_path):
        (tmp_path / "cleanup.py").write_text(SKILL_CODE)
        load_skill_registry(str(tmp_path))
        keys_to_clean = [k for k in sys.modules if k.startswith("user_skill_")]
        for k in keys_to_clean:
            del sys.modules[k]
