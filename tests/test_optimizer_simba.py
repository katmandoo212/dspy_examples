"""Tests for SIMBA optimizer."""

import pytest


def test_simba_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    assert optimizer.get_name() == "simba"


def test_simba_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    desc = optimizer.get_description()

    assert "SIMBA" in desc
    assert "stochastic" in desc.lower() or "mini-batch" in desc.lower()


def test_simba_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    config = optimizer.get_config()

    assert config.name == "simba"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_simba_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="simba",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = SIMBAOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_simba_inherits_from_base():
    """Test that SIMBAOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_simba_default_bsize():
    """Test that default batch size is set."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    assert optimizer.get_bsize() == 32


def test_simba_custom_bsize():
    """Test that batch size can be customized."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer(bsize=64)
    assert optimizer.get_bsize() == 64


def test_simba_default_max_steps():
    """Test that default max steps is set."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    assert optimizer.get_max_steps() == 8


def test_simba_custom_max_steps():
    """Test that max steps can be customized."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer(max_steps=12)
    assert optimizer.get_max_steps() == 12


def test_simba_custom_num_candidates():
    """Test that num_candidates can be customized."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer(num_candidates=10)
    assert optimizer._num_candidates == 10


def test_simba_custom_max_demos():
    """Test that max_demos can be customized."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer(max_demos=8)
    assert optimizer._max_demos == 8


def test_simba_prompt_model():
    """Test that prompt_model can be configured."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer()
    assert optimizer._prompt_model is None

    optimizer_with_model = SIMBAOptimizer(prompt_model="gpt-4")
    assert optimizer_with_model._prompt_model == "gpt-4"


def test_simba_temperature_settings():
    """Test that temperature settings can be customized."""
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = SIMBAOptimizer(
        temperature_for_sampling=0.5,
        temperature_for_candidates=0.3,
    )
    assert optimizer._temperature_for_sampling == 0.5
    assert optimizer._temperature_for_candidates == 0.3