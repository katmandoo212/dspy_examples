"""Tests for PromptOptimizer base interface."""

import pytest


def test_optimization_result_defaults():
    """Test OptimizationResult has correct defaults."""
    from dspy_examples.optimizers.base import OptimizationResult

    result = OptimizationResult(
        optimized_prompt="Test prompt",
        optimizer_name="test",
        provider_name="ollama",
        original_length=100,
        optimized_length=150,
    )

    assert result.optimized_prompt == "Test prompt"
    assert result.optimizer_name == "test"
    assert result.provider_name == "ollama"
    assert result.original_length == 100
    assert result.optimized_length == 150
    assert result.optimization_time == 0.0
    assert result.metadata == {}
    assert result.cache_key is None


def test_optimizer_config_defaults():
    """Test OptimizerConfig has correct defaults."""
    from dspy_examples.optimizers.base import OptimizerConfig

    config = OptimizerConfig(name="test_optimizer")

    assert config.name == "test_optimizer"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3
    assert config.num_threads == 4
    assert config.extra == {}


def test_prompt_optimizer_is_abstract():
    """Test that PromptOptimizer cannot be instantiated directly."""
    from dspy_examples.optimizers.base import PromptOptimizer

    with pytest.raises(TypeError):
        PromptOptimizer()


def test_prompt_optimizer_abstract_methods():
    """Test that PromptOptimizer has required abstract methods."""
    from dspy_examples.optimizers.base import PromptOptimizer
    import inspect

    abstract_methods = {
        name
        for name, method in inspect.getmembers(PromptOptimizer)
        if getattr(method, "__isabstractmethod__", False)
    }

    assert "optimize" in abstract_methods
    assert "get_name" in abstract_methods
    assert "get_description" in abstract_methods
    assert "get_config" in abstract_methods