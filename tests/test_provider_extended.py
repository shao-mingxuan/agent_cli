"""Anthropic / DeepSeek Provider 测试。"""

import pytest

import agent_cli.providers.anthropic as anthropic_mod
from agent_cli.cli.factory import create_provider
from agent_cli.providers.anthropic import AnthropicProvider
from agent_cli.providers.base import BaseProvider
from agent_cli.providers.deepseek import DeepSeekProvider
from agent_cli.providers.openai_compat import OpenAICompatProvider


class TestAnthropicProvider:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        provider = AnthropicProvider()
        assert provider.api_key is None
        assert provider.base_url is None
        assert provider.model_name == "claude-sonnet-4-5"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        provider = AnthropicProvider()
        assert provider.api_key == "env-key"
        assert provider.model_name == "claude-3-5-sonnet-latest"

    def test_init_override(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicProvider(
            api_key="custom-key", model_name="claude-opus-4-1", base_url="http://localhost"
        )
        assert provider.api_key == "custom-key"
        assert provider.model_name == "claude-opus-4-1"
        assert provider.base_url == "http://localhost"

    def test_get_model_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicProvider()
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            provider.get_model()

    def test_get_model_missing_dep(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(anthropic_mod, "ChatAnthropic", None)
        provider = AnthropicProvider()
        with pytest.raises(ValueError, match="langchain-anthropic"):
            provider.get_model()

    def test_get_model_config(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(anthropic_mod, "ChatAnthropic", FakeChatAnthropic)
        provider = AnthropicProvider()
        model = provider.get_model()
        assert model.kwargs["model"] == provider.model_name
        assert model.kwargs["api_key"] == "test-key"
        assert model.kwargs["streaming"] is True
        assert model.kwargs["max_tokens"] == 4096

    def test_switch_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(anthropic_mod, "ChatAnthropic", FakeChatAnthropic)
        provider = AnthropicProvider()
        provider.switch_model("claude-3-5-haiku-latest")
        assert provider.model_name == "claude-3-5-haiku-latest"

    def test_no_embeddings(self):
        assert AnthropicProvider(api_key="x").get_embeddings() is None


class TestDeepSeekProvider:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        provider = DeepSeekProvider()
        assert provider.base_url == "https://api.deepseek.com"
        assert provider.model_name == "deepseek-chat"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        provider = DeepSeekProvider()
        assert provider.api_key == "env-key"
        assert provider.model_name == "deepseek-reasoner"

    def test_init_override(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = DeepSeekProvider(
            api_key="custom-key", model_name="deepseek-chat", base_url="http://localhost"
        )
        assert provider.api_key == "custom-key"
        assert provider.base_url == "http://localhost"

    def test_get_model_missing_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = DeepSeekProvider()
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            provider.get_model()

    def test_get_model_config(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        provider = DeepSeekProvider()
        model = provider.get_model()
        assert model.model_name == "deepseek-chat"
        assert "api.deepseek.com" in model.openai_api_base

    def test_switch_model(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        provider = DeepSeekProvider()
        provider.switch_model("deepseek-reasoner")
        assert provider.model_name == "deepseek-reasoner"

    def test_no_embeddings(self):
        assert DeepSeekProvider(api_key="x").get_embeddings() is None


class TestCreateProviderExtended:
    def test_anthropic_name(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("anthropic"), AnthropicProvider)

    def test_anthropic_alias(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("claude"), AnthropicProvider)

    def test_deepseek_name(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("deepseek"), DeepSeekProvider)

    def test_deepseek_alias(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("ds"), DeepSeekProvider)

    def test_unknown_falls_back(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("unknown"), OpenAICompatProvider)

    def test_provider_env_anthropic(self, monkeypatch):
        monkeypatch.setenv("PROVIDER", "anthropic")
        assert isinstance(create_provider(), AnthropicProvider)

    def test_return_base_type(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("deepseek"), BaseProvider)
