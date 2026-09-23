"""L3 技能 - 技能注册中心。"""

from .define_skill import Skill


class SkillRegistry:
    """技能注册中心，管理所有可用技能。"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        return list(self._skills.keys())


def create_default_registry() -> SkillRegistry:
    """创建带有内置技能的注册中心。"""
    from .builtin.research import research_skill
    from .builtin.summarize_doc import summarize_doc_skill

    registry = SkillRegistry()
    registry.register(research_skill)
    registry.register(summarize_doc_skill)
    return registry
