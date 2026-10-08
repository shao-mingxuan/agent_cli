"""L5 模型客户端 - Provider 抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseProvider(ABC):
    """所有模型 Provider 的抽象基类。"""

    @abstractmethod
    def get_model(self) -> Any:
        """返回一个可用的 LangChain ChatModel 实例。"""
        ...

    def switch_model(self, model_name: str) -> Any:
        """切换当前模型，返回新的 ChatModel 实例。"""
        raise NotImplementedError("当前 Provider 不支持运行时切换模型")

    def get_embeddings(self) -> Callable[[str], list[float]] | None:
        """返回 embedding 回调函数，不支持时返回 None。"""
        return None
