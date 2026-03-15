"""Tests for BootstrapFewShot optimizer."""

import pytest


def test_bootstrap_fewshot_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer

    optimizer = BootstrapFewShotOptimizer()
    assert optimizer.get_name() == "bootstrap_fewshot"


def test_bootstrap_fewshot_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer

    optimizer = BootstrapFewShotOptimizer()
    desc = optimizer.get_description()

    assert "BootstrapFewShot" in desc
    assert "few-shot" in desc.lower()


def test_bootstrap_fewshot_get_config():
    """Test optimizer config."""
    from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer

    optimizer = BootstrapFewShotOptimizer()
    config = optimizer.get_config()

    assert config.name == "bootstrap_fewshot"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_bootstrap_fewshot_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="bootstrap_fewshot",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
    )
    optimizer = BootstrapFewShotOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10