"""Tests for OpenRouter provider adapter."""

import os
from unittest.mock import patch


def test_openrouter_provider_from_env():
    """Test creating OpenRouterProvider from environment."""
    from dspy_examples.providers.openrouter import OpenRouterProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {
        "OPENROUTER_API_KEY": "or-test-key",
        "OPENROUTER_MODEL": "anthropic/claude-3-opus",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        provider = OpenRouterProvider.from_env()

        assert provider.config.model_name == "anthropic/claude-3-opus"
        assert provider.config.api_key == "or-test-key"
        assert provider.provider_name() == "openrouter"

    _reset_settings()


def test_openrouter_provider_validate_config():
    """Test OpenRouterProvider config validation."""
    from dspy_examples.providers.openrouter import OpenRouterProvider
    from dspy_examples.providers.base import LMConfig

    valid_config = LMConfig(
        model_name="anthropic/claude-3-opus",
        provider="openrouter",
        api_key="valid-or-key",
    )
    provider = OpenRouterProvider(valid_config)
    assert provider.validate_config() is True

    invalid_config = LMConfig(
        model_name="anthropic/claude-3-opus",
        provider="openrouter",
        api_key=None,
    )
    provider = OpenRouterProvider(invalid_config)
    assert provider.validate_config() is False


def test_openrouter_provider_create_lm():
    """Test creating DSPy LM from OpenRouterProvider."""
    from dspy_examples.providers.openrouter import OpenRouterProvider
    from dspy_examples.providers.base import LMConfig
    import dspy

    config = LMConfig(
        model_name="anthropic/claude-3-opus",
        provider="openrouter",
        api_key="or-test-key",
    )
    provider = OpenRouterProvider(config)
    lm = provider.create_lm()

    assert isinstance(lm, dspy.LM)