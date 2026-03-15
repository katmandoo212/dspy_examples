"""Tests for Google provider adapter."""

import os
from unittest.mock import patch


def test_google_provider_from_env():
    """Test creating GoogleProvider from environment."""
    from dspy_examples.providers.google import GoogleProvider
    from dspy_examples.settings import _reset_settings

    _reset_settings()
    env_vars = {
        "GOOGLE_API_KEY": "google-test-key",
        "GOOGLE_MODEL": "gemini-1.5-pro",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        provider = GoogleProvider.from_env()

        assert provider.config.model_name == "gemini-1.5-pro"
        assert provider.config.api_key == "google-test-key"
        assert provider.provider_name() == "google"

    _reset_settings()


def test_google_provider_validate_config():
    """Test GoogleProvider config validation."""
    from dspy_examples.providers.google import GoogleProvider
    from dspy_examples.providers.base import LMConfig

    valid_config = LMConfig(
        model_name="gemini-pro",
        provider="google",
        api_key="valid-google-key",
    )
    provider = GoogleProvider(valid_config)
    assert provider.validate_config() is True

    invalid_config = LMConfig(
        model_name="gemini-pro",
        provider="google",
        api_key=None,
    )
    provider = GoogleProvider(invalid_config)
    assert provider.validate_config() is False


def test_google_provider_create_lm():
    """Test creating DSPy LM from GoogleProvider."""
    from dspy_examples.providers.google import GoogleProvider
    from dspy_examples.providers.base import LMConfig
    import dspy

    config = LMConfig(
        model_name="gemini-pro",
        provider="google",
        api_key="google-test-key",
    )
    provider = GoogleProvider(config)
    lm = provider.create_lm()

    assert isinstance(lm, dspy.LM)