"""Prompt optimization strategies (Strategy Pattern)."""

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig
from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer
from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer
from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer
from dspy_examples.optimizers.gepa import GEPAOptimizer
from dspy_examples.optimizers.better_together import BetterTogetherOptimizer
from dspy_examples.optimizers.copro import COPROOptimizer
from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer
from dspy_examples.optimizers.simba import SIMBAOptimizer

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "OptimizerConfig",
    "BootstrapFewShotOptimizer",
    "BootstrapRandomOptimizer",
    "MIPROv2Optimizer",
    "GEPAOptimizer",
    "BetterTogetherOptimizer",
    "COPROOptimizer",
    "BootstrapFinetuneOptimizer",
    "SIMBAOptimizer",
]