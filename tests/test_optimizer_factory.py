"""Tests for OptimizerFactory."""

import pytest


def test_optimizer_factory_list_optimizers():
    """Test listing available optimizers."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory

    optimizers = OptimizerFactory.list_optimizers()

    assert "bootstrap_fewshot" in optimizers
    assert "bootstrap_random" in optimizers
    assert "mipro_v2" in optimizers
    assert "gepa" in optimizers
    assert "better_together" in optimizers
    assert "copro" in optimizers
    assert "bootstrap_finetune" in optimizers
    assert "simba" in optimizers


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


def test_optimizer_factory_create_mipro_v2():
    """Test creating MIPROv2 optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer

    optimizer = OptimizerFactory.create("mipro_v2")

    assert isinstance(optimizer, MIPROv2Optimizer)
    assert optimizer.get_name() == "mipro_v2"
    assert optimizer.get_auto_mode() == "medium"


def test_optimizer_factory_create_gepa():
    """Test creating GEPA optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.gepa import GEPAOptimizer

    optimizer = OptimizerFactory.create("gepa")

    assert isinstance(optimizer, GEPAOptimizer)
    assert optimizer.get_name() == "gepa"
    assert optimizer.get_auto_mode() == "medium"


def test_optimizer_factory_create_better_together():
    """Test creating BetterTogether optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

    optimizer = OptimizerFactory.create("better_together")

    assert isinstance(optimizer, BetterTogetherOptimizer)
    assert optimizer.get_name() == "better_together"
    assert optimizer.get_strategy() == "p -> w"
    assert optimizer.get_prompt_optimizer() == "bootstrap_random"


def test_optimizer_factory_create_copro():
    """Test creating COPRO optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.copro import COPROOptimizer

    optimizer = OptimizerFactory.create("copro")

    assert isinstance(optimizer, COPROOptimizer)
    assert optimizer.get_name() == "copro"
    assert optimizer.get_breadth() == 10


def test_optimizer_factory_create_bootstrap_finetune():
    """Test creating BootstrapFinetune optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer

    optimizer = OptimizerFactory.create("bootstrap_finetune")

    assert isinstance(optimizer, BootstrapFinetuneOptimizer)
    assert optimizer.get_name() == "bootstrap_finetune"
    assert optimizer.get_multitask() is True


def test_optimizer_factory_create_simba():
    """Test creating SIMBA optimizer via factory."""
    from dspy_examples.factory.optimizer_factory import OptimizerFactory
    from dspy_examples.optimizers.simba import SIMBAOptimizer

    optimizer = OptimizerFactory.create("simba")

    assert isinstance(optimizer, SIMBAOptimizer)
    assert optimizer.get_name() == "simba"
    assert optimizer.get_bsize() == 32


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