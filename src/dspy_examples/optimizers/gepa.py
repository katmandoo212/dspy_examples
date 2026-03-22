"""GEPA (Genetic-Pareto) optimizer implementation.

GEPA is a reflective prompt optimizer that uses LLMs to analyze execution traces
and propose improved prompts. It maintains a Pareto frontier of candidates and
can leverage textual feedback, not just scalar scores.

Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
arXiv:2507.19457
"""

import dspy
from dspy.teleprompt import GEPA

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt using reflective evolution."""

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


class GEPAOptimizer(PromptOptimizer):
    """GEPA optimization strategy with reflective prompt evolution.

    GEPA uses LLM reflection to analyze execution traces and propose improved
    prompts. It maintains a Pareto frontier of candidates optimized across
    different evaluation instances.

    Attributes:
        config: Optimizer configuration.
        auto: Budget preset ('light', 'medium', 'heavy').
        reflection_model: Optional model for reflection (uses configured LM if None).
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        auto: str = "medium",
        reflection_model: str | None = None,
    ) -> None:
        """Initialize GEPA optimizer.

        Args:
            config: Optimizer configuration.
            auto: Budget preset - 'light' (quick), 'medium' (balanced), 'heavy' (thorough).
            reflection_model: Optional model for reflection. If None, uses the configured LM.
        """
        self._config = config or OptimizerConfig(name="gepa")
        self._auto = auto
        self._reflection_model = reflection_model

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using GEPA.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # GEPA metric signature: (gold, pred, trace, pred_name, pred_trace)
        # Returns score and optional textual feedback
        def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """Metric with textual feedback for GEPA reflection."""
            # Accept all predictions, provide feedback for reflection
            if hasattr(pred, 'optimized_prompt') and pred.optimized_prompt:
                score = 1.0
                feedback = f"Successfully generated optimized prompt (length: {len(pred.optimized_prompt)})"
            else:
                score = 0.0
                feedback = "Failed to generate optimized prompt"

            # Return ScoreWithFeedback dict for GEPA's reflective learning
            return {"score": score, "feedback": feedback}

        # Build GEPA optimizer
        optimizer_kwargs = {
            "metric": metric_with_feedback,
            "auto": self._auto,
        }

        # Add reflection LM if specified
        if self._reflection_model:
            optimizer_kwargs["reflection_lm"] = dspy.LM(model=self._reflection_model, temperature=1.0, max_tokens=32000)

        optimizer = GEPA(**optimizer_kwargs)

        module = PromptOptimizerModule()

        # Compile with training examples
        # Note: GEPA can also accept valset, but we use trainset for simplicity
        optimized_module = optimizer.compile(module, trainset=trainset)

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
                "auto": self._auto,
                "reflection_model": self._reflection_model or "default",
            },
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "gepa"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "GEPA optimizer using reflective prompt evolution with Pareto-based candidate selection."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config

    def get_auto_mode(self) -> str:
        """Return the auto mode setting."""
        return self._auto