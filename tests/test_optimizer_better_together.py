"""Tests for BetterTogether optimizer."""

import pytest


def test_better_together_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    assert optimizer.get_name() == "better_together"


def test_better_together_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    desc = optimizer.get_description()

    assert "BetterTogether" in desc
    assert "meta-optimizer" in desc.lower()


def test_better_together_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    config = optimizer.get_config()

    assert config.name == "better_together"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_better_together_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="better_together",
        max_bootstrapped_demos=5,
        max_labeled_demos=10,
        num_threads=8,
    )
    optimizer = BetterTogetherOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 5
    assert config.max_labeled_demos == 10
    assert config.num_threads == 8


def test_better_together_inherits_from_base():
    """Test that BetterTogetherOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_better_together_default_strategy():
    """Test that default strategy is 'p -> w'."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    assert optimizer.get_strategy() == "p -> w"


def test_better_together_custom_strategy():
    """Test that strategy can be customized."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer(strategy="w -> p")
    assert optimizer.get_strategy() == "w -> p"

    optimizer_full = BetterTogetherOptimizer(strategy="p -> w -> p")
    assert optimizer_full.get_strategy() == "p -> w -> p"


def test_better_together_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    with pytest.raises(ValueError, match="Invalid strategy"):
        BetterTogetherOptimizer(strategy="invalid")


def test_better_together_default_prompt_optimizer():
    """Test that default prompt optimizer is 'bootstrap_random'."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer()
    assert optimizer.get_prompt_optimizer() == "bootstrap_random"


def test_better_together_custom_prompt_optimizer():
    """Test that prompt optimizer can be customized."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = BetterTogetherOptimizer(prompt_optimizer="gepa")
    assert optimizer.get_prompt_optimizer() == "gepa"


def test_better_together_invalid_prompt_optimizer():
    """Test that invalid prompt optimizer raises ValueError."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    with pytest.raises(ValueError, match="Invalid prompt_optimizer"):
        BetterTogetherOptimizer(prompt_optimizer="invalid")


def test_better_together_valid_strategies():
    """Test all valid strategies are accepted."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    valid_strategies = ("p -> w", "w -> p", "p -> w -> p")
    for strategy in valid_strategies:
        optimizer = BetterTogetherOptimizer(strategy=strategy)
        assert optimizer.get_strategy() == strategy


def test_better_together_valid_prompt_optimizers():
    """Test all valid prompt optimizers are accepted."""
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    valid_optimizers = ("bootstrap_random", "gepa")
    for opt in valid_optimizers:
        optimizer = BetterTogetherOptimizer(prompt_optimizer=opt)
        assert optimizer.get_prompt_optimizer() == opt