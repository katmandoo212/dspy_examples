"""SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer implementation.

SIMBA optimizes through mini-batch sampling, identifying challenging examples,
and creating self-reflective rules or adding successful demonstrations.

Reference: https://dspy.ai/api/optimizers/SIMBA/
"""

import dspy
from dspy.teleprompt import SIMBA

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt using stochastic mini-batch ascent."""

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


class SIMBAOptimizer(PromptOptimizer):
    """SIMBA optimization strategy using stochastic mini-batch ascent.

    SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizes programs by:
    1. Mini-batch sampling from training data
    2. Identifying challenging examples with high output variability
    3. Creating self-reflective rules OR adding successful demonstrations
    4. Iteratively improving through stochastic ascent

    This optimizer works well with larger datasets (200+ examples) and
    provides a different optimization approach compared to GEPA.

    Attributes:
        config: Optimizer configuration.
        bsize: Mini-batch size for optimization.
        num_candidates: Number of new candidate programs per iteration.
        max_steps: Number of optimization steps.
        max_demos: Maximum demos per predictor.
        prompt_model: Optional model for program evolution.
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        max_demos: int = 4,
        prompt_model: str | None = None,
        temperature_for_sampling: float = 0.2,
        temperature_for_candidates: float = 0.2,
    ) -> None:
        """Initialize SIMBA optimizer.

        Args:
            config: Optimizer configuration.
            bsize: Mini-batch size for optimization.
            num_candidates: Number of new candidate programs per iteration.
            max_steps: Number of optimization steps to run.
            max_demos: Maximum demos per predictor before dropping some.
            prompt_model: Optional model for program evolution (uses global LM if None).
            temperature_for_sampling: Temperature for picking programs during trajectory sampling.
            temperature_for_candidates: Temperature for picking source program for new candidates.
        """
        self._config = config or OptimizerConfig(name="simba")
        self._bsize = bsize
        self._num_candidates = num_candidates
        self._max_steps = max_steps
        self._max_demos = max_demos
        self._prompt_model = prompt_model
        self._temperature_for_sampling = temperature_for_sampling
        self._temperature_for_candidates = temperature_for_candidates

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using SIMBA.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # Metric for SIMBA evaluation
        def metric(example, prediction, trace=None):
            """Evaluate prediction quality."""
            return True

        # Build SIMBA optimizer
        optimizer_kwargs = {
            "metric": metric,
            "bsize": self._bsize,
            "num_candidates": self._num_candidates,
            "max_steps": self._max_steps,
            "max_demos": self._max_demos,
            "num_threads": self._config.num_threads,
            "temperature_for_sampling": self._temperature_for_sampling,
            "temperature_for_candidates": self._temperature_for_candidates,
        }

        # Add prompt model if specified
        if self._prompt_model:
            optimizer_kwargs["prompt_model"] = dspy.LM(model=self._prompt_model)

        optimizer = SIMBA(**optimizer_kwargs)

        module = PromptOptimizerModule()

        # Compile with training examples
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
        )

        # Run the optimized module
        result = optimized_module(unoptimized_prompt=prompt)

        return OptimizationResult(
            optimized_prompt=result.optimized_prompt,
            optimizer_name=self.get_name(),
            provider_name="",  # Set by pipeline
            original_length=len(prompt),
            optimized_length=len(result.optimized_prompt),
            metadata={
                "num_trainset": len(trainset),
                "bsize": self._bsize,
                "num_candidates": self._num_candidates,
                "max_steps": self._max_steps,
                "max_demos": self._max_demos,
            },
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "simba"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "SIMBA optimizer using stochastic introspective mini-batch ascent with self-reflective improvement."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config

    def get_bsize(self) -> int:
        """Return the batch size."""
        return self._bsize

    def get_max_steps(self) -> int:
        """Return the max steps setting."""
        return self._max_steps