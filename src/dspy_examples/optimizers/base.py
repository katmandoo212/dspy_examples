"""Base interface for prompt optimizers (Strategy Pattern)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import dspy


@dataclass
class OptimizationResult:
    """Result of a prompt optimization."""

    optimized_prompt: str
    optimizer_name: str
    provider_name: str
    original_length: int
    optimized_length: int
    optimization_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""

    name: str
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 3
    num_threads: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


class PromptOptimizer(ABC):
    """Strategy interface for prompt optimizers."""

    @abstractmethod
    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using this technique.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return optimizer identifier (e.g., 'bootstrap_fewshot')."""
        ...

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description."""
        ...

    @abstractmethod
    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        ...