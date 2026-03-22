"""Tests for GEPA optimizer."""

import pytest


def test_gepa_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    assert optimizer.get_name() == "gepa"


def test_gepa_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    desc = optimizer.get_description()

    assert "GEPA" in desc
    assert "reflective" in desc.lower()


def test_gepa_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    config = optimizer.get_config()

    assert config.name == "gepa"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_gepa_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="gepa",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = GEPAOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_gepa_inherits_from_base():
    """Test that GEPAOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_gepa_auto_mode_default():
    """Test that default auto mode is set."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    assert optimizer.get_auto_mode() == "medium"


def test_gepa_auto_mode_custom():
    """Test that auto mode can be customized."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer(auto="light")
    assert optimizer.get_auto_mode() == "light"

    optimizer_heavy = GEPAOptimizer(auto="heavy")
    assert optimizer_heavy.get_auto_mode() == "heavy"


def test_gepa_reflection_model():
    """Test that reflection model can be configured."""
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    assert optimizer._reflection_model is None

    optimizer_with_model = GEPAOptimizer(reflection_model="gpt-4")
    assert optimizer_with_model._reflection_model == "gpt-4"