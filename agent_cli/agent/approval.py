"""L1 编排 - 审批/敏感工具检测 Helper 函数 + 常量。"""

from langchain_core.messages import ToolMessage

MAX_TOOL_FAILURES = 3
_TOOL_ERROR_MARKERS = ("error", "mcp error", "exception", "traceback", "failed")

# 敏感工具关键词 —— 按匹配策略分为两类：
#
# 1. _SENSITIVE_WORDS：按单词匹配（以下划线或空白分隔）。
#    避免短关键词在子串中误匹配（如 "sh" 命中 "push"、"db" 命中 "adblock"）。
# 2. _SENSITIVE_PATTERNS：按子串匹配，仅包含长度 >=5 的特定完整词组，
#    因长度足够，几乎不会产生误匹配。
_SENSITIVE_WORDS: set[str] = {
    "shell",
    "exec",
    "bash",
    "spawn",
    "command",
    "run",
    "sql",
    "database",
    "db",
    "insert",
    "update",
    "drop",
    "write",
    "delete",
    "remove",
    "create",
    "mkdir",
    "rmdir",
    "edit",
    "modify",
    "patch",
    "move",
    "copy",
    "rename",
    "chmod",
    "chown",
    "file",
    "directory",
    "git",
    "commit",
    "push",
    "pull",
    "fs",
}

_SENSITIVE_PATTERNS: set[str] = {
    "filesystem",
    "read_file",
    "write_file",
    "edit_file",
    "read_directory",
    "list_directory",
    "search_files",
    "fs_",
}


def is_tool_error(content: str) -> bool:
    """检测工具返回内容是否为错误。"""
    if not content:
        return False
    lower = content.lower()
    return any(marker in lower for marker in _TOOL_ERROR_MARKERS)


def is_sensitive_tool(tool_call: dict) -> bool:
    """判断工具调用是否为敏感操作（需要人工审批）。

    匹配策略：
    1. 单词匹配（以下划线或空白分隔）——防止短关键词在子串中误匹配
       例如 "sh" 不再命中 "push"，"db" 不再命中 "adblock"。
    2. 子串匹配 —— 仅对长度 >=5 的特定完整词组生效。
    """
    name = tool_call.get("name", "").lower()
    category = tool_call.get("category", "").lower()
    combined = f"{name} {category}"

    words = set(combined.replace("_", " ").split())
    if words & _SENSITIVE_WORDS:
        return True

    for pattern in _SENSITIVE_PATTERNS:
        if pattern in combined:
            return True

    return False


def builtin_approval_callback(tool_calls: list[dict]) -> list[bool]:
    """内置默认审批回调：非敏感自动通过，敏感自动拒绝（安全默认）。"""
    return [not tc.get("sensitive", False) for tc in tool_calls]


def extract_pending_tool_calls(orch) -> list[dict]:
    """从中断状态中提取待审批的工具调用。"""
    config = {"configurable": {"thread_id": orch._thread_id}}
    state = orch.agent.get_state(config)
    messages = state.values.get("messages", [])
    pending = []
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                info = orch._tool_lookup.get(name)
                pending.append(
                    {
                        "id": tc["id"],
                        "name": name,
                        "args": args,
                        "source": info.source if info else "unknown",
                        "category": info.category if info else "",
                        "sensitive": is_sensitive_tool(
                            {
                                "name": name,
                                "category": info.category if info else "",
                            }
                        ),
                    }
                )
            break
    return pending


def handle_rejection(orch, rejected_ids: list[str]) -> None:
    """对被拒绝的工具调用注入拒绝 ToolMessage 并更新状态。"""
    config = {"configurable": {"thread_id": orch._thread_id}}
    rejected_msgs = [
        ToolMessage(
            content="用户拒绝了此工具调用。",
            name="rejected",
            tool_call_id=tc_id,
        )
        for tc_id in rejected_ids
    ]
    if rejected_msgs:
        orch.agent.update_state(config, {"messages": rejected_msgs}, as_node="tools")
