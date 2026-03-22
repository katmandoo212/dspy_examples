"""Factory for creating optimizer instances."""

from dspy_examples.optimizers.base import PromptOptimizer, OptimizerConfig
from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer
from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer
from dspy_examples.optimizers.mipro_v2 import MIPROv2Optimizer
from dspy_examples.optimizers.gepa import GEPAOptimizer
from dspy_examples.optimizers.better_together import BetterTogetherOptimizer
from dspy_examples.optimizers.copro import COPROOptimizer
from dspy_examples.optimizers.bootstrap_finetune import BootstrapFinetuneOptimizer
from dspy_examples.optimizers.simba import SIMBAOptimizer


class OptimizerFactory:
    """Factory for creating optimizer instances."""

    _optimizers: dict[str, type[PromptOptimizer]] = {
        "bootstrap_fewshot": BootstrapFewShotOptimizer,
        "bootstrap_random": BootstrapRandomOptimizer,
        "mipro_v2": MIPROv2Optimizer,
        "gepa": GEPAOptimizer,
        "better_together": BetterTogetherOptimizer,
        "copro": COPROOptimizer,
        "bootstrap_finetune": BootstrapFinetuneOptimizer,
        "simba": SIMBAOptimizer,
    }

    @classmethod
    def create(cls, optimizer_name: str, config: OptimizerConfig | None = None) -> PromptOptimizer:
        """Create an optimizer instance by name.

        Args:
            optimizer_name: The optimizer identifier (e.g., 'bootstrap_fewshot').
            config: Optional optimizer configuration.

        Returns:
            A PromptOptimizer instance.

        Raises:
            ValueError: If optimizer_name is not recognized.
        """
        if optimizer_name not in cls._optimizers:
            available = ", ".join(cls._optimizers.keys())
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: {available}")

        optimizer_class = cls._optimizers[optimizer_name]

        if config:
            return optimizer_class(config=config)
        return optimizer_class()

    @classmethod
    def register(cls, name: str, optimizer_class: type[PromptOptimizer]) -> None:
        """Register a new optimizer class.

        Args:
            name: The optimizer identifier.
            optimizer_class: The PromptOptimizer subclass to register.
        """
        cls._optimizers[name] = optimizer_class

    @classmethod
    def list_optimizers(cls) -> list[str]:
        """List all registered optimizer names.

        Returns:
            List of optimizer identifiers.
        """
        return list(cls._optimizers.keys())

    @classmethod
    def auto_select(cls, prompt: str) -> PromptOptimizer:
        """Auto-select best optimizer based on prompt characteristics.

        Args:
            prompt: The prompt to optimize.

        Returns:
            A PromptOptimizer instance.
        """
        # Use bootstrap_random for longer prompts (more complex optimization)
        # Use bootstrap_fewshot for shorter prompts (faster)
        if len(prompt) > 1000:
            return cls.create("bootstrap_random")
        return cls.create("bootstrap_fewshot")