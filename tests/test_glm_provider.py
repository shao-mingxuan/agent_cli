"""GLM Provider 与 Provider 选择逻辑测试。"""

import pytest

from agent_cli.cli.factory import create_provider
from agent_cli.providers.base import BaseProvider
from agent_cli.providers.glm import ZhipuProvider
from agent_cli.providers.openai_compat import OpenAICompatProvider


class TestZhipuProvider:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_API_BASE", raising=False)
        monkeypatch.delenv("GLM_MODEL", raising=False)
        provider = ZhipuProvider()
        assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"
        assert provider.model_name == "glm-4-flash"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        monkeypatch.setenv("GLM_MODEL", "glm-4.5")
        provider = ZhipuProvider()
        assert provider.api_key == "test-key"
        assert provider.model_name == "glm-4.5"

    def test_init_override(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        provider = ZhipuProvider(api_key="custom-key", model_name="glm-4-air", base_url="http://localhost")
        assert provider.api_key == "custom-key"
        assert provider.model_name == "glm-4-air"
        assert provider.base_url == "http://localhost"

    def test_get_model_missing_key(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        provider = ZhipuProvider()
        with pytest.raises(ValueError, match="ZHIPUAI_API_KEY"):
            provider.get_model()

    def test_get_model_config(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        provider = ZhipuProvider()
        model = provider.get_model()
        assert model.model_name == provider.model_name
        assert model.openai_api_key.get_secret_value() == "test-key"
        assert provider.base_url in model.openai_api_base

    def test_embeddings_missing_key(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        provider = ZhipuProvider()
        assert provider.get_embeddings() is None


class TestCreateProvider:
    def test_default_openai_compat(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider(), OpenAICompatProvider)

    def test_glm_name(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("glm"), ZhipuProvider)

    def test_glm_alias(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("zhipuai"), ZhipuProvider)

    def test_provider_env(self, monkeypatch):
        monkeypatch.setenv("PROVIDER", "glm")
        assert isinstance(create_provider(), ZhipuProvider)

    def test_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PROVIDER", "glm")
        assert isinstance(create_provider("openai_compat"), OpenAICompatProvider)

    def test_returns_base_type(self, monkeypatch):
        monkeypatch.delenv("PROVIDER", raising=False)
        assert isinstance(create_provider("glm"), BaseProvider)
