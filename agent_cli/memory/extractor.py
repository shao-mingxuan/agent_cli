"""L4 记忆 - 事实提取器：从对话中提取可持久化的事实。"""

from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

FACT_EXTRACTION_PROMPT = (
    "请从以下对话中提取用户透露的持久性事实（personal facts），"
    "即跨会话有用的信息，如：用户偏好、项目背景、技术栈选择、"
    "重要决策、常用环境配置等。每条事实用简洁的一句话描述。"
    "如果对话中没有值得持久化的事实，返回空行。"
    "格式：每行一条事实，不要编号。"
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
