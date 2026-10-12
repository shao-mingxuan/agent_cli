"""L5 模型客户端 - Anthropic（Claude）Provider。"""

from __future__ import annotations

import os

from .base import BaseProvider

# LLM 请求超时（秒），与其他 Provider 保持一致，避免 API 挂起时阻塞对话。
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# langchain-anthropic 为可选依赖，缺失时提示安装而非 import 时报错。
try:  # pragma: no cover - 依赖探测
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

_INSTALL_HINT = "pip install langchain-anthropic"


class AnthropicProvider(BaseProvider):
    """Claude 系列模型 Provider，走 Anthropic 官方 API。

    依赖 langchain-anthropic（可选），未安装时 get_model() 抛出带安装提示的
    ValueError。Claude 不提供 embedding API，get_embeddings() 恒返回 None。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        self.model_name = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.timeout = REQUEST_TIMEOUT

    def get_model(self) -> ChatAnthropic:
        if not self.api_key:
            raise ValueError("未配置 ANTHROPIC_API_KEY，请检查 .env 文件")
        if ChatAnthropic is None:
            raise ValueError(f"缺少 langchain-anthropic 依赖，请执行: {_INSTALL_HINT}")
        return ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            streaming=True,
            max_tokens=4096,
        )

    def switch_model(self, model_name: str) -> ChatAnthropic:
        """切换模型并返回新的 ChatAnthropic 实例。"""
        self.model_name = model_name
        return self.get_model()

    def get_embeddings(self) -> None:
        """Claude 无 embedding API，返回 None。"""
