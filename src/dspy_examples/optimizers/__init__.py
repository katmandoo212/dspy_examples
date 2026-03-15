"""Prompt optimization strategies (Strategy Pattern)."""

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig
from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer

__all__ = ["PromptOptimizer", "OptimizationResult", "OptimizerConfig", "BootstrapFewShotOptimizer"]