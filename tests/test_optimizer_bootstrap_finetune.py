"""Tests for BootstrapFinetune optimizer."""

import pytest


def test_bootstrap_finetune_get_name():
    """Test optimizer name."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    assert optimizer.get_name() == "bootstrap_finetune"


def test_bootstrap_finetune_get_description():
    """Test optimizer description."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    desc = optimizer.get_description()

    assert "BootstrapFinetune" in desc or "fine-tuning" in desc.lower()


def test_bootstrap_finetune_get_config_defaults():
    """Test optimizer config defaults."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    config = optimizer.get_config()

    assert config.name == "bootstrap_finetune"
    assert config.max_bootstrapped_demos == 3
    assert config.max_labeled_demos == 3


def test_bootstrap_finetune_custom_config():
    """Test optimizer with custom config."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="bootstrap_finetune",
        num_threads=8,
    )
    optimizer = BootstrapFinetuneOptimizer(config=custom_config)
    config = optimizer.get_config()

    assert config.num_threads == 8


def test_bootstrap_finetune_inherits_from_base():
    """Test that BootstrapFinetuneOptimizer inherits from PromptOptimizer."""
    from dspy_examples.optimizers.base import PromptOptimizer
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    assert isinstance(optimizer, PromptOptimizer)


def test_bootstrap_finetune_multitask_default():
    """Test that multitask defaults to True."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    assert optimizer.get_multitask() is True


def test_bootstrap_finetune_multitask_custom():
    """Test that multitask can be disabled."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer(multitask=False)
    assert optimizer.get_multitask() is False


def test_bootstrap_finetune_train_kwargs_default():
    """Test that default train_kwargs are set."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    kwargs = optimizer.get_train_kwargs()

    assert "learning_rate" in kwargs
    assert "num_train_epochs" in kwargs
    assert kwargs["learning_rate"] == 5e-5


def test_bootstrap_finetune_custom_train_kwargs():
    """Test that train_kwargs can be customized."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    custom_kwargs = {
        "learning_rate": 1e-5,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
    }
    optimizer = BootstrapFinetuneOptimizer(train_kwargs=custom_kwargs)
    kwargs = optimizer.get_train_kwargs()

    assert kwargs["learning_rate"] == 1e-5
    assert kwargs["num_train_epochs"] == 5
    assert kwargs["per_device_train_batch_size"] == 8


def test_bootstrap_finetune_train_kwargs_returns_copy():
    """Test that get_train_kwargs returns a copy."""
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = BootstrapFinetuneOptimizer()
    kwargs1 = optimizer.get_train_kwargs()
    kwargs2 = optimizer.get_train_kwargs()

    # Should be equal but not the same object
    assert kwargs1 == kwargs2
    assert kwargs1 is not kwargs2