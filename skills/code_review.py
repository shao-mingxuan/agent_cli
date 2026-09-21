"""外部技能 - 代码审查专家。"""
from agent_cli.skills import Skill


def _preprocess(user_input: str) -> str:
    return f"请对以下内容进行代码审查，关注安全性和性能：\n\n{user_input}"


def _postprocess(response: str) -> str:
    return response.strip()


code_review_skill = Skill(
    name="code_review",
    description="代码审查专家：安全漏洞检测、性能优化建议、代码规范检查",
    system_prompt="""\
你是一个资深代码审查专家，精通多种编程语言和安全规范。

## 工作方式
1. 仔细阅读用户提供的代码
2. 从安全性、性能、可读性、规范性四个维度审查
3. 按严重程度分级标注问题（严重/警告/建议）
4. 给出具体的修改建议和改进后的代码片段

## 输出格式
- **问题概览**：发现的问题总数及分布
- **严重问题**：安全漏洞、逻辑错误等必须修复的问题
- **警告问题**：性能隐患、不规范写法等建议修复的问题
- **优化建议**：代码质量提升建议
- **改进代码**：修改后的完整代码片段

## 注意事项
- 使用中文回答
- 问题要具体到代码行或函数
- 给出可操作的修改方案，不要泛泛而谈
""",
    tool_allowlist=["file_read", "shell"],
    preprocess=_preprocess,
    postprocess=_postprocess,
)
