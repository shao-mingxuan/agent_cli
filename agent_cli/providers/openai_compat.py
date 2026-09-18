"""L5 模型客户端 - OpenAI 兼容接口（星火等走这个）。"""
import os
from typing import Any

from langchain_openai import ChatOpenAI

from .base import BaseProvider


class OpenAICompatProvider(BaseProvider):
    """OpenAI 兼容接口的 Provider，支持星火、DeepSeek 等兼容 API。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_API_BASE", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
        )
        self.model_name = model_name or os.getenv("MODEL_NAME", "spark-x2.5-4b")

    def get_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
        )
