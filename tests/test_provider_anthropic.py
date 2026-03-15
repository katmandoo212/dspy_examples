"""Tests for Anthropic provider adapter."""

import os
from unittest.mock import patch

import pytest


def test_anthropic_provider_from_env():
    """Test creating AnthropicProvider from environment."""
    from dspy_examples.providers.anthropic import AnthropicProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        provider = AnthropicProvider.from_env()

        assert provider.config.model_name == "claude-3-sonnet-20240229"
        assert provider.config.api_key == "sk-ant-test-key"
        assert provider.provider_name() == "anthropic"

    _reset_settings()


def test_anthropic_provider_validate_config():
    """Test AnthropicProvider config validation."""
    from dspy_examples.providers.anthropic import AnthropicProvider
    from dspy_examples.providers.base import LMConfig

    valid_config = LMConfig(
        model_name="claude-3-opus-20240229",
        provider="anthropic",
        api_key="sk-ant-valid-key",
    )
    provider = AnthropicProvider(valid_config)
    assert provider.validate_config() is True

    invalid_config = LMConfig(
        model_name="claude-3-opus-20240229",
        provider="anthropic",
        api_key=None,
    )
    provider = AnthropicProvider(invalid_config)
    assert provider.validate_config() is False


def test_anthropic_provider_create_lm():
    """Test creating DSPy LM from AnthropicProvider."""
    from dspy_examples.providers.anthropic import AnthropicProvider
    from dspy_examples.providers.base import LMConfig
    import dspy

    config = LMConfig(
        model_name="claude-3-opus-20240229",
        provider="anthropic",
        api_key="sk-ant-test-key",
    )
    provider = AnthropicProvider(config)
    lm = provider.create_lm()

    assert isinstance(lm, dspy.LM)