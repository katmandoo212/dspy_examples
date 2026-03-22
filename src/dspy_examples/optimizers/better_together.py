"""BetterTogether optimizer implementation.

BetterTogether is a meta-optimizer that combines prompt optimization and
weight optimization (fine-tuning) in configurable sequences.

Paper: "Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together"
by Dilara Soylu, Christopher Potts, Omar Khattab.
"""

import dspy
from dspy.teleprompt import BetterTogether, BootstrapFinetune, BootstrapFewShotWithRandomSearch

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt using combined prompt and weight optimization."""

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


class BetterTogetherOptimizer(PromptOptimizer):
    """BetterTogether meta-optimization strategy.

    Combines prompt optimization (p) and weight optimization (w) in configurable
    sequences. The insight is that prompt optimization discovers effective strategies
    while weight optimization specializes the model.

    Common strategies:
        - "p -> w": Optimize prompts first, then fine-tune
        - "w -> p": Fine-tune first, then optimize prompts
        - "p -> w -> p": Full cycle (default)

    Attributes:
        config: Optimizer configuration.
        prompt_optimizer: Which prompt optimizer to use ('bootstrap_random' or 'gepa').
        strategy: Optimization sequence (default: "p -> w").
        auto: Budget preset for GEPA if used as prompt optimizer.
    """

    # Valid prompt optimizers that can be used with BetterTogether
    VALID_PROMPT_OPTIMIZERS = ("bootstrap_random", "gepa")

    # Valid optimization strategies
    VALID_STRATEGIES = ("p -> w", "w -> p", "p -> w -> p")

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        prompt_optimizer: str = "bootstrap_random",
        strategy: str = "p -> w",
        auto: str = "medium",
    ) -> None:
        """Initialize BetterTogether optimizer.

        Args:
            config: Optimizer configuration.
            prompt_optimizer: Prompt optimizer to use ('bootstrap_random' or 'gepa').
            strategy: Optimization sequence ('p -> w', 'w -> p', or 'p -> w -> p').
            auto: Budget preset for GEPA if used as prompt optimizer.

        Raises:
            ValueError: If prompt_optimizer or strategy is invalid.
        """
        if prompt_optimizer not in self.VALID_PROMPT_OPTIMIZERS:
            valid = ", ".join(self.VALID_PROMPT_OPTIMIZERS)
            raise ValueError(f"Invalid prompt_optimizer '{prompt_optimizer}'. Valid: {valid}")

        if strategy not in self.VALID_STRATEGIES:
            valid = ", ".join(self.VALID_STRATEGIES)
            raise ValueError(f"Invalid strategy '{strategy}'. Valid: {valid}")

        self._config = config or OptimizerConfig(name="better_together")
        self._prompt_optimizer = prompt_optimizer
        self._strategy = strategy
        self._auto = auto

    def _create_prompt_optimizer(self):
        """Create the prompt optimizer instance."""
        # Metric for prompt optimizers
        def metric(example, prediction, trace=None):
            """Accept all bootstrapped demonstrations."""
            return True

        if self._prompt_optimizer == "gepa":
            from dspy_examples.optimizers.gepa import GEPAOptimizer
            # For BetterTogether, we need the raw DSPy optimizer, not our wrapper
            return dspy.GEPA(
                metric=metric,
                auto=self._auto,
            )
        else:
            # Default: BootstrapFewShotWithRandomSearch
            return BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=self._config.max_bootstrapped_demos,
                max_labeled_demos=self._config.max_labeled_demos,
                num_threads=self._config.num_threads,
            )

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using BetterTogether.

        Note: BetterTogether requires fine-tuning support from the LM provider.
        Not all providers support BootstrapFinetune (requires fine-tuning API).

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # Metric for weight optimizer (BootstrapFinetune)
        def metric(example, prediction, trace=None):
            """Accept all for training."""
            return True

        # Create optimizers
        p_optimizer = self._create_prompt_optimizer()
        w_optimizer = BootstrapFinetune(metric=metric)

        # Create BetterTogether meta-optimizer
        optimizer = BetterTogether(
            metric=metric,
            p=p_optimizer,
            w=w_optimizer,
        )

        module = PromptOptimizerModule()

        # Ensure LM is set for fine-tuning
        # This uses the currently configured DSPy LM
        lm = dspy.settings.get("lm", default=None)
        if lm:
            module.set_lm(lm)

        # Compile with training examples
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
            strategy=self._strategy,
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
                "prompt_optimizer": self._prompt_optimizer,
                "strategy": self._strategy,
            },
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "better_together"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "BetterTogether meta-optimizer combining prompt and weight optimization for compound improvements."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config

    def get_prompt_optimizer(self) -> str:
        """Return the prompt optimizer name."""
        return self._prompt_optimizer

    def get_strategy(self) -> str:
        """Return the optimization strategy."""
        return self._strategy