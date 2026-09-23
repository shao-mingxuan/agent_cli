"""L5 模型客户端 - OpenAI 兼容接口（星火等走这个）。"""

import os
from typing import Callable

from langchain_openai import ChatOpenAI

from .base import BaseProvider

# LLM / embedding 请求超时（秒），超时抛错触发降级，
# 避免 API 挂起时阻塞对话或退出流程。
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))


class OpenAICompatProvider(BaseProvider):
    """OpenAI 兼容接口的 Provider，支持星火、DeepSeek 等兼容 API。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_API_BASE", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
        )
        self.model_name = model_name or os.getenv("MODEL_NAME", "spark-x2.5-4b")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-ada-002"
        )
        self.timeout = REQUEST_TIMEOUT

    def get_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def get_embeddings(self) -> Callable[[str], list[float]] | None:
        """创建 embedding 回调，API key 缺失或 import 失败时返回 None。"""
        if not self.api_key:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

            def _embed(text: str) -> list[float]:
                return embeddings.embed_query(text)

            return _embed
        except Exception:
            return None
