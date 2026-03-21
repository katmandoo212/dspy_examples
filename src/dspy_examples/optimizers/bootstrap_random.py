"""BootstrapFewShotWithRandomSearch optimizer implementation."""

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt by adding few-shot examples."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt with examples")


class PromptOptimizerModule(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


class BootstrapRandomOptimizer(PromptOptimizer):
    """BootstrapFewShotWithRandomSearch optimization strategy.

    This optimizer uses random search over demonstration combinations,
    which can find better few-shot selections than deterministic BootstrapFewShot.
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self._config = config or OptimizerConfig(name="bootstrap_random")

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using BootstrapFewShotWithRandomSearch.

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

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self._config.max_bootstrapped_demos,
            max_labeled_demos=self._config.max_labeled_demos,
            num_threads=self._config.num_threads,
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
            metadata={"num_trainset": len(trainset)},
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "bootstrap_random"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "BootstrapFewShotWithRandomSearch optimizer that uses random search to find optimal few-shot demonstration combinations."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config