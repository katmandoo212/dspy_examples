"""Tests for OpenAI provider adapter."""

import os
from unittest.mock import patch

import pytest


def test_openai_provider_from_env():
    """Test creating OpenAIProvider from environment."""
    from dspy_examples.providers.openai import OpenAIProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {
        "OPENAI_API_KEY": "sk-test-key",
        "OPENAI_MODEL": "gpt-4-turbo",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        provider = OpenAIProvider.from_env()

        assert provider.config.model_name == "gpt-4-turbo"
        assert provider.config.api_key == "sk-test-key"
        assert provider.provider_name() == "openai"

    _reset_settings()


def test_openai_provider_validate_config():
    """Test OpenAIProvider config validation."""
    from dspy_examples.providers.openai import OpenAIProvider
    from dspy_examples.providers.base import LMConfig

    valid_config = LMConfig(
        model_name="gpt-4",
        provider="openai",
        api_key="sk-valid-key",
    )
    provider = OpenAIProvider(valid_config)
    assert provider.validate_config() is True

    invalid_config = LMConfig(
        model_name="gpt-4",
        provider="openai",
        api_key=None,
    )
    provider = OpenAIProvider(invalid_config)
    assert provider.validate_config() is False


def test_openai_provider_create_lm():
    """Test creating DSPy LM from OpenAIProvider."""
    from dspy_examples.providers.openai import OpenAIProvider
    from dspy_examples.providers.base import LMConfig
    import dspy

    config = LMConfig(
        model_name="gpt-4",
        provider="openai",
        api_key="sk-test-key",
    )
    provider = OpenAIProvider(config)
    lm = provider.create_lm()

    assert isinstance(lm, dspy.LM)