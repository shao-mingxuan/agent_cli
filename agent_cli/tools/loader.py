"""L3 工具 - Python 插件加载器。"""

import importlib.util
import os
import sys

from langchain_core.tools import BaseTool

from .registry import ToolInfo


def load_plugins(plugins_dir: str = "./plugins") -> list[ToolInfo]:
    """扫描目录下的 .py 文件，收集 @tool 装饰的函数。

    Args:
        plugins_dir: 插件目录路径

    Returns:
        ToolInfo 列表，source="plugin"，category=文件名（不含 .py）
    """
    infos: list[ToolInfo] = []

    if not os.path.isdir(plugins_dir):
        return infos

    for filename in sorted(os.listdir(plugins_dir)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue

        filepath = os.path.join(plugins_dir, filename)
        module_name = f"plugin_{filename[:-3]}"
        category = filename[:-3]

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[插件加载警告] {filename}: {e}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BaseTool):
                infos.append(
                    ToolInfo(
                        tool=attr,
                        name=attr.name,
                        source="plugin",
                        category=category,
                    )
                )

    return infos
