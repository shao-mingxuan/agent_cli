"""L5 模型客户端 - Provider 抽象基类。"""
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """所有模型 Provider 的抽象基类。"""

    @abstractmethod
    def get_model(self) -> Any:
        """返回一个可用的 LangChain ChatModel 实例。"""
        ...
