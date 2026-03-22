"""MIPROv2 optimizer implementation."""

import dspy
from dspy.teleprompt import MIPROv2

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt using MIPROv2."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt")


class PromptOptimizerModule(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


class MIPROv2Optimizer(PromptOptimizer):
    """MIPROv2 optimization strategy with Bayesian instruction optimization."""

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        auto_mode: str = "medium",
    ) -> None:
        """Initialize MIPROv2 optimizer.

        Args:
            config: Optimizer configuration.
            auto_mode: Auto mode setting ('light', 'medium', 'heavy').
        """
        self._config = config or OptimizerConfig(name="mipro_v2")
        self._auto_mode = auto_mode

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using MIPROv2.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # Create a simple metric that accepts all bootstrapped demos
        def metric(example, prediction, trace=None):
            """Accept all bootstrapped demonstrations."""
            return True

        optimizer = MIPROv2(
            metric=metric,
            max_bootstrapped_demos=self._config.max_bootstrapped_demos,
            max_labeled_demos=self._config.max_labeled_demos,
            num_threads=self._config.num_threads,
            auto=self._auto_mode,
        )

        module = PromptOptimizerModule()

        # Compile with training examples
        optimized_module = optimizer.compile(module, trainset=trainset)

        # Run the optimized module
        result = optimized_module(unoptimized_prompt=prompt)

        return OptimizationResult(
            optimized_prompt=result.optimized_prompt,
            optimizer_name=self.get_name(),
            provider_name="",  # Set by pipeline
            original_length=len(prompt),
            optimized_length=len(result.optimized_prompt),
            metadata={"num_trainset": len(trainset), "auto_mode": self._auto_mode},
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "mipro_v2"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "MIPROv2 optimizer that uses Bayesian optimization to find optimal instructions and few-shot examples."

    def get_config(self) -> OptimizerConfig:
        """Return optimizer configuration."""
        return self._config

    def get_auto_mode(self) -> str:
        """Return the auto mode setting."""
        return self._auto_mode