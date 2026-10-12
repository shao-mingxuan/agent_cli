"""L5 模型客户端 - DeepSeek Provider。"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .base import BaseProvider

# LLM 请求超时（秒），与其他 Provider 保持一致，避免 API 挂起时阻塞对话。
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))


class DeepSeekProvider(BaseProvider):
    """DeepSeek 系列模型 Provider，走 OpenAI 兼容接口。

    DeepSeek 官方提供 OpenAI 兼容端点（https://api.deepseek.com），
    直接复用 ChatOpenAI，无需额外依赖。deepseek-chat / deepseek-reasoner
    均可通过 model_name 切换。DeepSeek 无官方 embedding API，
    get_embeddings() 恒返回 None。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = REQUEST_TIMEOUT

    def get_model(self) -> ChatOpenAI:
        if not self.api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY，请检查 .env 文件")
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            streaming=True,
        )

    def switch_model(self, model_name: str) -> ChatOpenAI:
        """切换模型并返回新的 ChatOpenAI 实例。"""
        self.model_name = model_name
        return self.get_model()

    def get_embeddings(self) -> None:
        """DeepSeek 无 embedding API，返回 None。"""
