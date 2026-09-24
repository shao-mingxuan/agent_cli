"""L5 模型客户端 - GLM（智谱 AI）Provider。"""

import os
from typing import Callable

from langchain_openai import ChatOpenAI

from .base import BaseProvider

# LLM / embedding 请求超时（秒），与 OpenAI 兼容 Provider 保持一致，
# 避免 API 挂起时阻塞对话或退出流程。
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))


class ZhipuProvider(BaseProvider):
    """GLM 系列模型 Provider，走智谱 AI 的 OpenAI 兼容接口。

    API 文档：https://open.bigmodel.cn/，兼容 OpenAI SDK / ChatOpenAI。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
        )
        self.model_name = model_name or os.getenv("GLM_MODEL", "glm-4-flash")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "embedding-3"
        )
        self.timeout = REQUEST_TIMEOUT

    def get_model(self) -> ChatOpenAI:
        if not self.api_key:
            raise ValueError("未配置 ZHIPUAI_API_KEY，请检查 .env 文件")
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
