"""L3 技能 - 技能加载器。

支持从外部 Python 模块加载自定义 Skill。
"""

import importlib.util
import os
import sys

from .define_skill import Skill
from .registry import SkillRegistry, create_default_registry


def load_skill_modules(skills_dir: str = "./skills") -> list[Skill]:
    """扫描目录下的 .py 文件，收集 Skill 对象。

    Args:
        skills_dir: 技能模块目录路径

    Returns:
        Skill 列表
    """
    skills: list[Skill] = []

    if not os.path.isdir(skills_dir):
        return skills

    for filename in sorted(os.listdir(skills_dir)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue

        filepath = os.path.join(skills_dir, filename)
        module_name = f"user_skill_{filename[:-3]}"

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[技能加载警告] {filename}: {e}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, Skill):
                skills.append(attr)

    return skills


def load_skill_registry(skills_dir: str = "./skills") -> SkillRegistry:
    """加载内置技能 + 外部技能，返回注册中心。"""
    registry = create_default_registry()

    for skill in load_skill_modules(skills_dir):
        registry.register(skill)

    return registry
