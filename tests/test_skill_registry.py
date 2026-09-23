"""L3 SkillRegistry + define_skill 测试。"""

from agent_cli.skills.define_skill import Skill, define_skill
from agent_cli.skills.registry import SkillRegistry, create_default_registry


class TestSkill:
    def test_defaults(self):
        s = Skill(name="t", description="d", system_prompt="p")
        assert s.tool_allowlist is None
        assert s.preprocess is None
        assert s.postprocess is None

    def test_with_all_fields(self):
        s = Skill(
            name="t",
            description="d",
            system_prompt="p",
            tool_allowlist=["a", "b"],
            preprocess=lambda x: x,
            postprocess=lambda x: x,
        )
        assert s.tool_allowlist == ["a", "b"]
        assert s.preprocess is not None
        assert s.postprocess is not None

    def test_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(
            Skill(name="t", description="d", system_prompt="p")
        )


class TestSkillRegistry:
    def test_register_single_skill(self):
        reg = SkillRegistry()
        s = Skill(name="t", description="d", system_prompt="p")
        reg.register(s)
        assert reg.get("t") is s

    def test_get_nonexistent(self):
        assert SkillRegistry().get("nope") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="d", system_prompt="p"))
        reg.register(Skill(name="b", description="d", system_prompt="p"))
        assert len(reg.list_skills()) == 2

    def test_list_names(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="d", system_prompt="p"))
        reg.register(Skill(name="b", description="d", system_prompt="p"))
        assert set(reg.list_names()) == {"a", "b"}

    def test_list_skills_returns_copy(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="d", system_prompt="p"))
        skills = reg.list_skills()
        skills.clear()
        assert len(reg.list_skills()) == 1

    def test_register_overwrites_same_name(self):
        reg = SkillRegistry()
        s1 = Skill(name="t", description="d1", system_prompt="p")
        s2 = Skill(name="t", description="d2", system_prompt="p")
        reg.register(s1)
        reg.register(s2)
        assert reg.get("t").description == "d2"


class TestCreateDefaultRegistry:
    def test_has_2_skills(self):
        reg = create_default_registry()
        assert len(reg.list_skills()) == 2

    def test_skill_names(self):
        reg = create_default_registry()
        names = reg.list_names()
        assert "research" in names
        assert "summarize_doc" in names

    def test_research_has_prompt(self):
        reg = create_default_registry()
        research = reg.get("research")
        assert research.system_prompt is not None
        assert len(research.system_prompt) > 0

    def test_research_no_allowlist(self):
        reg = create_default_registry()
        research = reg.get("research")
        assert research.tool_allowlist is None

    def test_summarize_doc_has_prompt(self):
        reg = create_default_registry()
        skill = reg.get("summarize_doc")
        assert skill.system_prompt is not None
        assert len(skill.system_prompt) > 0


class TestDefineSkill:
    def test_returns_skill_instance(self):
        @define_skill("t", "d", "p")
        def my_skill():
            pass

        assert isinstance(my_skill, Skill)
        assert my_skill.name == "t"
        assert my_skill.description == "d"
        assert my_skill.system_prompt == "p"

    def test_inherits_preprocess_postprocess(self):
        def my_func():
            pass

        my_func.preprocess = lambda x: x.upper()
        my_func.postprocess = lambda x: x.strip()
        skill = define_skill("t", "d", "p")(my_func)
        assert skill.preprocess is not None
        assert skill.postprocess is not None

    def test_no_preprocess_postprocess(self):
        @define_skill("t", "d", "p")
        def my_skill():
            pass

        assert my_skill.preprocess is None
        assert my_skill.postprocess is None

    def test_with_tool_allowlist(self):
        @define_skill("t", "d", "p", tool_allowlist=["a"])
        def my_skill():
            pass

        assert my_skill.tool_allowlist == ["a"]
