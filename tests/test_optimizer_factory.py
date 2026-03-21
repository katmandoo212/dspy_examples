"""Tests for OptimizerFactory."""

import pytest


def test_optimizer_factory_list_optimizers():
    """Test listing available optimizers."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory

    optimizers = OptimizerFactory.list_optimizers()

    assert "bootstrap_fewshot" in optimizers
    assert "bootstrap_random" in optimizers


def test_optimizer_factory_create_bootstrap_fewshot():
    """Test creating BootstrapFewShot optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer

    optimizer = OptimizerFactory.create("bootstrap_fewshot")

    assert isinstance(optimizer, BootstrapFewShotOptimizer)
    assert optimizer.get_name() == "bootstrap_fewshot"


def test_optimizer_factory_create_bootstrap_random():
    """Test creating BootstrapRandom optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

    optimizer = OptimizerFactory.create("bootstrap_random")

    assert isinstance(optimizer, BootstrapRandomOptimizer)
    assert optimizer.get_name() == "bootstrap_random"


def test_optimizer_factory_create_with_config():
    """Test creating optimizer with custom config."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.base import OptimizerConfig

    custom_config = OptimizerConfig(
        name="bootstrap_fewshot",
        max_bootstrapped_demos=10,
    )
    optimizer = OptimizerFactory.create("bootstrap_fewshot", config=custom_config)
    config = optimizer.get_config()

    assert config.max_bootstrapped_demos == 10


def test_optimizer_factory_unknown_optimizer():
    """Test creating unknown optimizer raises error."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory

    with pytest.raises(ValueError, match="Unknown optimizer"):
        OptimizerFactory.create("unknown_optimizer")


def test_optimizer_factory_register_custom_optimizer():
    """Test registering a custom optimizer."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.base import PromptOptimizer, OptimizerConfig, OptimizationResult
    import dspy

    class CustomOptimizer(PromptOptimizer):
        def __init__(self, config: OptimizerConfig | None = None):
            self._config = config or OptimizerConfig(name="custom")

        def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
            return OptimizationResult(
                optimized_prompt=prompt,
                optimizer_name="custom",
                provider_name="",
                original_length=len(prompt),
                optimized_length=len(prompt),
            )

        def get_name(self) -> str:
            return "custom"

        def get_description(self) -> str:
            return "Custom optimizer"

        def get_config(self) -> OptimizerConfig:
            return self._config

    OptimizerFactory.register("custom", CustomOptimizer)

    assert "custom" in OptimizerFactory.list_optimizers()
    optimizer = OptimizerFactory.create("custom")
    assert isinstance(optimizer, CustomOptimizer)