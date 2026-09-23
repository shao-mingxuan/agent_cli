"""L1 编排 - 配置信息收集（从 orchestrator 提取）。"""

from typing import Any

from .approval import builtin_approval_callback


def build_config_info(orch) -> dict[str, Any]:
    """构建当前 Agent 配置的摘要信息。"""
    model = getattr(orch.provider, "model_name", "unknown")
    base_url = getattr(orch.provider, "base_url", "unknown")

    tools_by_source: dict[str, list[str]] = {}
    for name, info in orch._tool_lookup.items():
        tools_by_source.setdefault(info.source, []).append(name)

    mcp_tools: list[dict] = []
    if orch._mcp_registry:
        for info in orch._mcp_registry.get_tool_infos():
            mcp_tools.append(
                {
                    "name": info.name,
                    "category": info.category,
                }
            )

    skill_info = None
    if orch._active_skill:
        skill_info = {
            "name": orch._active_skill.name,
            "description": orch._active_skill.description,
            "tool_allowlist": orch._active_skill.tool_allowlist,
        }

    return {
        "model": model,
        "base_url": base_url,
        "system_prompt_preview": orch.system_prompt[:200]
        + ("..." if len(orch.system_prompt) > 200 else ""),
        "total_tools": len(orch._tool_lookup),
        "tools_by_source": tools_by_source,
        "mcp_tools": mcp_tools,
        "skill": skill_info,
        "approval_mode": (
            "full"
            if orch._approval_callback is not builtin_approval_callback
            else "sensitive"
        ),
        "max_messages": orch.memory.max_messages,
        "max_tokens": orch.memory.max_tokens,
        "compression_enabled": orch.memory.compression_enabled,
        "long_term_memory": orch._ltm_enabled,
        "episodic_count": (
            orch._episodic_memory.count() if orch._episodic_memory else 0
        ),
        "semantic_count": (
            orch._semantic_memory.count() if orch._semantic_memory else 0
        ),
    }
