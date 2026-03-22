"""Tests for COPRO optimizer."""

import pytest


def test_copro_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    assert optimizer.get_name() == "copro"


def test_copro_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    desc = optimizer.get_description()

    assert "COPRO" in desc
    assert "coordinate ascent" in desc.lower() or "instruction" in desc.lower()


def test_copro_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    config = optimizer.get_config()

    assert config.name == "copro"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_copro_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.copro import COPROOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="copro",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = COPROOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_copro_inherits_from_base():
    """Test that COPROOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_copro_default_breadth():
    """Test that default breadth is set."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    assert optimizer.get_breadth() == 10


def test_copro_custom_breadth():
    """Test that breadth can be customized."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer(breadth=20)
    assert optimizer.get_breadth() == 20


def test_copro_default_depth():
    """Test that default depth is set."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    assert optimizer.get_depth() == 3


def test_copro_custom_depth():
    """Test that depth can be customized."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer(depth=5)
    assert optimizer.get_depth() == 5


def test_copro_invalid_breadth():
    """Test that breadth <= 1 raises ValueError."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    with pytest.raises(ValueError, match="breadth must be greater than 1"):
        COPROOptimizer(breadth=1)

    with pytest.raises(ValueError, match="breadth must be greater than 1"):
        COPROOptimizer(breadth=0)


def test_copro_custom_temperature():
    """Test that init_temperature can be customized."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer(init_temperature=1.0)
    assert optimizer._init_temperature == 1.0


def test_copro_prompt_model():
    """Test that prompt_model can be configured."""
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = COPROOptimizer()
    assert optimizer._prompt_model is None

    optimizer_with_model = COPROOptimizer(prompt_model="gpt-4")
    assert optimizer_with_model._prompt_model == "gpt-4"