"""L4 记忆 - 事实提取器：从对话中提取可持久化的事实。"""

from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

FACT_EXTRACTION_PROMPT = (
    "你是记忆提取器，请从以下对话中提取值得跨会话记住的【用户】持久性事实。\n"
    "重点包括：用户的姓名或称呼、年龄、职业与岗位、技术栈偏好（框架/语言/工具）、\n"
    "兴趣爱好、项目背景、重要决策、常用环境配置等。\n"
    "规则：\n"
    "1. 只提取用户自己透露的事实，忽略助手的发言。\n"
    "2. 每条事实用第三人称描述，主语统一用“用户”。例如：用户的名字是张三；"
    "用户是前端工程师。\n"
    "3. 每条一行，用简洁的一句话，不要编号、不要多余的说明。\n"
    "4. 如果对话中没有值得持久化的信息，输出空行。\n"
)


def create_fact_extractor(model) -> Callable[[str], list[str]]:
    """用 LangChain ChatModel 创建事实提取回调。

    Args:
        model: LangChain ChatModel 实例，需支持 .invoke()。

    Returns:
        extractor 回调函数，输入对话文本，返回事实列表。
    """

    def _extract(text: str) -> list[str]:
        response = model.invoke(
            [
                SystemMessage(content=FACT_EXTRACTION_PROMPT),
                HumanMessage(content=text),
            ]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        facts = [line.strip() for line in content.strip().split("\n") if line.strip()]
        return facts

    return _extract
