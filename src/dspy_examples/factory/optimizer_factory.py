"""Factory for creating optimizer instances."""

from dspy_examples.optimizers.base import PromptOptimizer, OptimizerConfig
from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer


class OptimizerFactory:
    """Factory for creating optimizer instances."""

    _optimizers: dict[str, type[PromptOptimizer]] = {
        "bootstrap_fewshot": BootstrapFewShotOptimizer,
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
        # Simple heuristic: use bootstrap_fewshot for prompts < 1000 chars
        if len(prompt) < 1000:
            return cls.create("bootstrap_fewshot")
        return cls.create("bootstrap_fewshot")