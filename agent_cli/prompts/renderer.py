"""L2 提示词 - {{variable}} 模板替换。"""

import re

_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def render_template(template: str, variables: dict | None = None) -> str:
    """将 {{name}} 占位符替换为 variables 中的值。

    未提供的变量保持原样，便于即时替换与后续扩展。
    """
    if not template:
        return template
    variables = variables or {}

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        value = variables.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return _PATTERN.sub(_replace, template)


def render_prompt(template: str, **variables: object) -> str:
    return render_template(template, variables)
