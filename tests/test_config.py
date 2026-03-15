"""Tests for DSPy configuration module."""

import os
from unittest.mock import patch

import pytest


class TestLoadConfig:
    """Tests for loading DSPy configuration."""

    def test_load_config_returns_model_name(self) -> None:
        """Test that load_config returns the configured model name."""
        # Clear only the specific env vars we want to test defaults for
        env_patch = patch.dict(
            os.environ,
            {},
            clear=False
        )
        # Remove OLLAMA_MODEL to test default
        model_value = os.environ.pop("OLLAMA_MODEL", None)
        base_url_value = os.environ.pop("OLLAMA_BASE_URL", None)

        try:
            from dspy_examples.config import load_config

            config = load_config()
            assert "model" in config
            assert config["model"] == "gpt-oss:120b-cloud"
        finally:
            # Restore original values
            if model_value is not None:
                os.environ["OLLAMA_MODEL"] = model_value
            if base_url_value is not None:
                os.environ["OLLAMA_BASE_URL"] = base_url_value

    def test_load_config_returns_base_url(self) -> None:
        """Test that load_config returns the configured base URL."""
        # Remove env vars to test defaults
        model_value = os.environ.pop("OLLAMA_MODEL", None)
        base_url_value = os.environ.pop("OLLAMA_BASE_URL", None)

        try:
            from dspy_examples.config import load_config

            config = load_config()
            assert "base_url" in config
            assert config["base_url"] == "http://localhost:11434"
        finally:
            # Restore original values
            if model_value is not None:
                os.environ["OLLAMA_MODEL"] = model_value
            if base_url_value is not None:
                os.environ["OLLAMA_BASE_URL"] = base_url_value


class TestConfigureDspy:
    """Tests for configuring DSPy with Ollama."""

    @patch("dspy.configure")
    def test_configure_dspy_sets_lm(self, mock_configure) -> None:
        """Test that configure_dspy sets up the language model."""
        # Remove env vars to test defaults
        model_value = os.environ.pop("OLLAMA_MODEL", None)
        base_url_value = os.environ.pop("OLLAMA_BASE_URL", None)

        try:
            from dspy_examples.config import configure_dspy

            lm = configure_dspy()
            assert lm is not None
            assert lm.model == "ollama_chat/gpt-oss:120b-cloud"
            mock_configure.assert_called_once()
        finally:
            # Restore original values
            if model_value is not None:
                os.environ["OLLAMA_MODEL"] = model_value
            if base_url_value is not None:
                os.environ["OLLAMA_BASE_URL"] = base_url_value