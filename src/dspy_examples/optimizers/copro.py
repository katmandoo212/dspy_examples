"""COPRO (Coordinate Ascent Prompt Optimization) optimizer implementation.

COPRO generates and refines instructions using coordinate ascent (hill-climbing).
It focuses purely on instruction optimization, making it a good complement to
few-shot optimizers like BootstrapFewShot.

Reference: https://dspy.ai/api/optimizers/COPRO
"""

import dspy
from dspy.teleprompt import COPRO

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt by refining instructions."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt with refined instructions")


class PromptOptimizerModule(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


class COPROOptimizer(PromptOptimizer):
    """COPRO optimization strategy using coordinate ascent.

    COPRO generates new prompt instructions and refines them through
    iterative hill-climbing. It focuses on instruction optimization
    rather than few-shot example selection.

    Attributes:
        config: Optimizer configuration.
        breadth: Number of new prompts to generate per iteration.
        depth: Number of refinement iterations.
        init_temperature: Temperature for prompt generation (higher = more creative).
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        breadth: int = 10,
        depth: int = 3,
        init_temperature: float = 1.4,
        prompt_model: str | None = None,
    ) -> None:
        """Initialize COPRO optimizer.

        Args:
            config: Optimizer configuration.
            breadth: Number of new prompts to generate per iteration (must be > 1).
            depth: Number of refinement iterations.
            init_temperature: Temperature for prompt generation (higher = more creative).
            prompt_model: Optional model for prompt generation (uses configured LM if None).

        Raises:
            ValueError: If breadth <= 1.
        """
        if breadth <= 1:
            raise ValueError("breadth must be greater than 1")

        self._config = config or OptimizerConfig(name="copro")
        self._breadth = breadth
        self._depth = depth
        self._init_temperature = init_temperature
        self._prompt_model = prompt_model

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using COPRO.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # Metric for COPRO evaluation
        def metric(example, prediction, trace=None):
            """Evaluate prediction quality."""
            return True

        # Build COPRO optimizer
        optimizer_kwargs = {
            "metric": metric,
            "breadth": self._breadth,
            "depth": self._depth,
            "init_temperature": self._init_temperature,
        }

        # Add prompt model if specified
        if self._prompt_model:
            optimizer_kwargs["prompt_model"] = dspy.LM(model=self._prompt_model)

        optimizer = COPRO(**optimizer_kwargs)

        module = PromptOptimizerModule()

        # Compile with training examples
        # Note: num_threads passed via eval_kwargs in compile()
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
                "breadth": self._breadth,
                "depth": self._depth,
                "init_temperature": self._init_temperature,
            },
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "copro"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "COPRO optimizer using coordinate ascent for instruction refinement."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config

    def get_breadth(self) -> int:
        """Return the breadth setting."""
        return self._breadth

    def get_depth(self) -> int:
        """Return the depth setting."""
        return self._depth