"""L4 记忆 - 摘要器：将 LLM model 包装为 summarizer 回调。"""

from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

SUMMARY_SYSTEM_PROMPT = (
    "请将以下对话历史压缩为简洁的摘要。保留关键信息：用户意图、"
    "重要决策、工具调用结果的核心结论。用要点列表格式输出，"
    "不超过 300 字。"
)


def create_llm_summarizer(model) -> Callable[[str], str]:
    """用 LangChain ChatModel 创建摘要回调。

    Args:
        model: LangChain ChatModel 实例（如 ChatOpenAI），需支持 .invoke()。

    Returns:
        summarizer 回调函数，输入对话文本，返回摘要文本。
    """

    def _summarize(text: str) -> str:
        response = model.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ]
        )
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    return _summarize
