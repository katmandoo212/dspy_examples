"""Tests for Pydantic Settings configuration."""

import os
from unittest.mock import patch

import pytest


def test_settings_defaults():
    """Test that Settings has correct default values."""
    from dspy_examples.settings import Settings

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()

        assert settings.llm_provider == "ollama"
        assert settings.ollama_model == "gpt-oss:120b-cloud"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.optimizer == "bootstrap_fewshot"
        assert settings.use_cache is True
        assert settings.cache_dir == ".cache/optimizations"


def test_settings_from_env():
    """Test that Settings loads from environment variables."""
    from dspy_examples.settings import Settings

    env_vars = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key-123",
        "OPENAI_MODEL": "gpt-4-turbo",
        "OPTIMIZER": "bootstrap_random",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

        assert settings.llm_provider == "openai"
        assert settings.openai_api_key == "test-key-123"
        assert settings.openai_model == "gpt-4-turbo"
        assert settings.optimizer == "bootstrap_random"


def test_get_settings_singleton():
    """Test that get_settings returns a singleton instance."""
    from dspy_examples.settings import Settings, get_settings, _reset_settings

    _reset_settings()
    with patch.dict(os.environ, {}, clear=True):
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    _reset_settings()