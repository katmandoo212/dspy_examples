"""Tests for Ollama provider adapter."""

import os
from unittest.mock import patch

import pytest


def test_ollama_provider_from_env():
    """Test creating OllamaProvider from environment."""
    from dspy_examples.providers.ollama import OllamaProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {
        "OLLAMA_MODEL": "llama3",
        "OLLAMA_BASE_URL": "http://custom:11434",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        provider = OllamaProvider.from_env()

        assert provider.config.model_name == "llama3"
        assert provider.config.base_url == "http://custom:11434"
        assert provider.provider_name() == "ollama"

    _reset_settings()


def test_ollama_provider_validate_config():
    """Test OllamaProvider config validation."""
    from dspy_examples.providers.ollama import OllamaProvider
    from dspy_examples.providers.base import LMConfig

    valid_config = LMConfig(
        model_name="llama3",
        provider="ollama",
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(valid_config)
    assert provider.validate_config() is True

    invalid_config = LMConfig(
        model_name="",
        provider="ollama",
        base_url="",
    )
    provider = OllamaProvider(invalid_config)
    assert provider.validate_config() is False


def test_ollama_provider_create_lm():
    """Test creating DSPy LM from OllamaProvider."""
    from dspy_examples.providers.ollama import OllamaProvider
    from dspy_examples.providers.base import LMConfig
    import dspy

    config = LMConfig(
        model_name="llama3",
        provider="ollama",
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(config)
    lm = provider.create_lm()

    assert isinstance(lm, dspy.LM)
    assert lm.model == "ollama_chat/llama3"