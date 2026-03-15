"""Tests for DSPy configuration module."""

import os
from unittest.mock import patch

import pytest


class TestLoadConfig:
    """Tests for loading DSPy configuration."""

    def test_load_config_returns_model_name(self) -> None:
        """Test that load_config returns the configured model name."""
        from dspy_examples.config import load_config

        config = load_config()
        assert "model" in config
        assert config["model"] == "gpt-oss:120b-cloud"

    def test_load_config_returns_base_url(self) -> None:
        """Test that load_config returns the configured base URL."""
        from dspy_examples.config import load_config

        config = load_config()
        assert "base_url" in config
        assert config["base_url"] == "http://localhost:11434"


class TestConfigureDspy:
    """Tests for configuring DSPy with Ollama."""

    def test_configure_dspy_sets_lm(self) -> None:
        """Test that configure_dspy sets up the language model."""
        from dspy_examples.config import configure_dspy

        lm = configure_dspy()
        assert lm is not None
        assert lm.model == "ollama_chat/gpt-oss:120b-cloud"