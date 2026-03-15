"""Tests for LMProvider base interface."""

import pytest


def test_lmconfig_defaults():
    """Test LMConfig has correct defaults."""
    from dspy_examples.providers.base import LMConfig

    config = LMConfig(model_name="test-model", provider="test")

    assert config.model_name == "test-model"
    assert config.provider == "test"
    assert config.api_key is None
    assert config.base_url is None
    assert config.temperature == 0.0
    assert config.max_tokens is None
    assert config.extra == {}


def test_lmconfig_custom_values():
    """Test LMConfig with custom values."""
    from dspy_examples.providers.base import LMConfig

    config = LMConfig(
        model_name="gpt-4",
        provider="openai",
        api_key="secret-key",
        temperature=0.7,
        max_tokens=1000,
        extra={"top_p": 0.9},
    )

    assert config.model_name == "gpt-4"
    assert config.provider == "openai"
    assert config.api_key == "secret-key"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.extra == {"top_p": 0.9}


def test_lmprovider_is_abstract():
    """Test that LMProvider cannot be instantiated directly."""
    from dspy_examples.providers.base import LMProvider

    with pytest.raises(TypeError):
        LMProvider()


def test_lmprovider_abstract_methods():
    """Test that LMProvider has required abstract methods."""
    from dspy_examples.providers.base import LMProvider
    import inspect

    abstract_methods = {
        name
        for name, method in inspect.getmembers(LMProvider)
        if getattr(method, "__isabstractmethod__", False)
    }

    assert "create_lm" in abstract_methods
    assert "validate_config" in abstract_methods
    assert "from_env" in abstract_methods
    assert "provider_name" in abstract_methods