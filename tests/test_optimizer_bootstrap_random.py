"""Tests for BootstrapRandom optimizer."""

import pytest


def test_bootstrap_random_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

    optimizer = BootstrapRandomOptimizer()
    assert optimizer.get_name() == "bootstrap_random"


def test_bootstrap_random_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

    optimizer = BootstrapRandomOptimizer()
    desc = optimizer.get_description()

    assert "BootstrapFewShotWithRandomSearch" in desc
    assert "random" in desc.lower()


def test_bootstrap_random_get_config():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

    optimizer = BootstrapRandomOptimizer()
    config = optimizer.get_config()

    assert config.name == "bootstrap_random"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_bootstrap_random_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="bootstrap_random",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = BootstrapRandomOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_bootstrap_random_inherits_from_base():
    """Test that BootstrapRandomOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

    optimizer = BootstrapRandomOptimizer()
    assert isinstance(optimizer, PromptOptimizer)