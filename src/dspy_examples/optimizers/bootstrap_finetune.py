"""BootstrapFinetune optimizer implementation.

BootstrapFinetune distills prompt-based programs into weight updates
for fine-tuning LLMs. It requires fine-tuning support from the LM provider.

Reference: https://dspy.ai/api/optimizers/BootstrapFinetune
"""

import dspy
from dspy.teleprompt import BootstrapFinetune

from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig


class PromptOptimization(dspy.Signature):
    """Optimize a prompt through fine-tuning."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt after fine-tuning")


class PromptOptimizerModule(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


class BootstrapFinetuneOptimizer(PromptOptimizer):
    """BootstrapFinetune optimization strategy.

    Distills prompt-based programs into weight updates for fine-tuning.
    This optimizer requires fine-tuning support from the LM provider
    (e.g., OpenAI, Databricks, Local with fine-tuning capabilities).

    Note:
        This optimizer requires an LM with fine-tuning support.
        Not all providers support fine-tuning (e.g., Ollama, Anthropic).

    Attributes:
        config: Optimizer configuration.
        multitask: Whether to share training data across predictors.
        train_kwargs: Training hyperparameters for fine-tuning.
    """

    # Default training arguments for fine-tuning
    DEFAULT_TRAIN_KWARGS = {
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        multitask: bool = True,
        train_kwargs: dict | None = None,
    ) -> None:
        """Initialize BootstrapFinetune optimizer.

        Args:
            config: Optimizer configuration.
            multitask: Whether to share training data across predictors (default: True).
            train_kwargs: Training hyperparameters (uses defaults if None).
        """
        self._config = config or OptimizerConfig(name="bootstrap_finetune")
        self._multitask = multitask
        self._train_kwargs = train_kwargs or self.DEFAULT_TRAIN_KWARGS.copy()

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using BootstrapFinetune.

        Note:
            This method requires fine-tuning support. It will fail if the
            configured LM provider does not support fine-tuning.

        Args:
            prompt: The unoptimized prompt text.
            trainset: Training examples for optimization.

        Returns:
            The optimization result with optimized prompt.
        """
        # Metric for filtering traces (accept all by default)
        def metric(example, prediction, trace=None):
            """Accept all traces for fine-tuning."""
            return True

        # Create BootstrapFinetune optimizer
        optimizer = BootstrapFinetune(
            metric=metric,
            multitask=self._multitask,
            train_kwargs=self._train_kwargs,
            num_threads=self._config.num_threads,
        )

        module = PromptOptimizerModule()

        # Ensure LM is set for fine-tuning
        lm = dspy.settings.get("lm", default=None)
        if lm:
            module.set_lm(lm)

        # Compile with training examples
        try:
            optimized_module = optimizer.compile(
                module,
                trainset=trainset,
            )
        except Exception as e:
            # Provide helpful error message for unsupported providers
            if "fine-tuning" in str(e).lower() or "finetune" in str(e).lower():
                raise NotImplementedError(
                    f"Fine-tuning not supported by the current LM provider. "
                    f"BootstrapFinetune requires an LM with fine-tuning capabilities "
                    f"(e.g., OpenAI, Databricks, Local). Current error: {e}"
                ) from e
            raise

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
                "multitask": self._multitask,
                "train_kwargs": self._train_kwargs,
            },
        )

    def get_name(self) -> str:
        """Return optimizer identifier."""
        return "bootstrap_finetune"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "BootstrapFinetune optimizer that distills prompts into weight updates for fine-tuning."

    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        return self._config

    def get_multitask(self) -> bool:
        """Return the multitask setting."""
        return self._multitask

    def get_train_kwargs(self) -> dict:
        """Return the training hyperparameters."""
        return self._train_kwargs.copy()