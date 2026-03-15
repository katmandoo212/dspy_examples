"""Tests for ProviderFactory."""

import os
from unittest.mock import patch

import pytest


def test_provider_factory_list_providers():
    """Test listing available providers."""
    from dspy_examples.factory.provider_factory import ProviderFactory

    providers = ProviderFactory.list_providers()

    assert "ollama" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "openrouter" in providers


def test_provider_factory_create_ollama():
    """Test creating Ollama provider via factory."""
    from dspy_examples.factory.provider_factory import ProviderFactory
    from dspy_examples.providers.ollama import OllamaProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    with patch.dict(os.environ, {}, clear=True):
        provider = ProviderFactory.create("ollama")

        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name() == "ollama"

    _reset_settings()


def test_provider_factory_create_openai():
    """Test creating OpenAI provider via factory."""
    from dspy_examples.factory.provider_factory import ProviderFactory
    from dspy_examples.providers.openai import OpenAIProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {"OPENAI_API_KEY": "sk-test-key"}

    with patch.dict(os.environ, env_vars, clear=True):
        provider = ProviderFactory.create("openai")

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name() == "openai"

    _reset_settings()


def test_provider_factory_unknown_provider():
    """Test creating unknown provider raises error."""
    from dspy_examples.factory.provider_factory import ProviderFactory

    with pytest.raises(ValueError, match="Unknown provider"):
        ProviderFactory.create("unknown_provider")


def test_provider_factory_register_custom_provider():
    """Test registering a custom provider."""
    from dspy_examples.factory.provider_factory import ProviderFactory
    from dspy_examples.providers.base import LMProvider, LMConfig
    import dspy

    class CustomProvider(LMProvider):
        def __init__(self, config: LMConfig):
            self.config = config

        @classmethod
        def from_env(cls) -> "CustomProvider":
            return cls(LMConfig(model_name="custom-model", provider="custom"))

        def create_lm(self) -> dspy.LM:
            return dspy.LM(model="custom/model")

        def validate_config(self) -> bool:
            return True

        @classmethod
        def provider_name(cls) -> str:
            return "custom"

    ProviderFactory.register("custom", CustomProvider)

    assert "custom" in ProviderFactory.list_providers()
    provider = ProviderFactory.create("custom")
    assert isinstance(provider, CustomProvider)