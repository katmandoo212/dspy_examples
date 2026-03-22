"""Tests for MIPROv2 optimizer."""

import pytest


def test_mipro_v2_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer()
    assert optimizer.get_name() == "mipro_v2"


def test_mipro_v2_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer()
    desc = optimizer.get_description()

    assert "MIPROv2" in desc
    assert "instruction" in desc.lower() or "bayesian" in desc.lower()


def test_mipro_v2_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer()
    config = optimizer.get_config()

    assert config.name == "mipro_v2"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_mipro_v2_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="mipro_v2",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = MIPROv2Optimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_mipro_v2_inherits_from_base():
    """Test that MIPROv2Optimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_mipro_v2_auto_mode_default():
    """Test that default auto mode is set."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer()
    assert optimizer.get_auto_mode() == "medium"


def test_mipro_v2_auto_mode_custom():
    """Test that auto mode can be customized."""
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = MIPROv2Optimizer(auto_mode="light")
    assert optimizer.get_auto_mode() == "light"

    optimizer_heavy = MIPROv2Optimizer(auto_mode="heavy")
    assert optimizer_heavy.get_auto_mode() == "heavy"