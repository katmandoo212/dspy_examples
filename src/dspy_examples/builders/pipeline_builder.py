"""PipelineBuilder for fluent configuration.

Usage:
    pipeline = (PipelineBuilder()
        .with_prompt("prompt.md")
        .with_provider("openai", model="gpt-4")
        .with_optimizer("mipro_v2")
        .with_observer(LoggingObserver())
        .build())
"""

from __future__ import annotations

from pathlib import Path

from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
from dspy_examples.observers import Observer


class PipelineBuilder:
    """Builder for OptimizationPipeline with fluent API.

    Provides method chaining for pipeline configuration.

    Example:
        pipeline = (PipelineBuilder()
            .with_prompt("prompt.md")
            .with_provider("openai")
            .with_optimizer("mipro_v2")
            .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._config = PipelineConfig()
        self._observers: list[Observer] = []

    def with_prompt(self, path: str | Path) -> "PipelineBuilder":
        """Set input prompt path.

        Args:
            path: Path to prompt file

        Returns:
            Self for method chaining
        """
        self._config.input_path = Path(path)
        return self

    def with_output(self, path: str | Path) -> "PipelineBuilder":
        """Set output path.

        Args:
            path: Path for output file

        Returns:
            Self for method chaining
        """
        self._config.output_path = Path(path)
        return self

    def with_provider(self, name: str, model: str | None = None) -> "PipelineBuilder":
        """Set LLM provider.

        Args:
            name: Provider name (e.g., "openai", "ollama", "anthropic")
            model: Optional model name (currently unused, reserved for future use)

        Returns:
            Self for method chaining

        Note:
            Model configuration is currently handled via environment variables.
            The model parameter is accepted for API consistency but not yet used.
            Future implementation may store model name in PipelineConfig.
        """
        self._config.provider_name = name
        # TODO: Store model name in config when PipelineConfig supports it
        _ = model  # Acknowledge parameter to satisfy linters
        return self

    def with_optimizer(self, name: str) -> "PipelineBuilder":
        """Set optimizer.

        Args:
            name: Optimizer name (e.g., "bootstrap_fewshot", "mipro_v2")

        Returns:
            Self for method chaining
        """
        self._config.optimizer_name = name
        return self

    def with_variables(self, variables: dict[str, str]) -> "PipelineBuilder":
        """Set template variables.

        Args:
            variables: Dictionary of variable names to values

        Returns:
            Self for method chaining
        """
        self._config.variables = variables
        return self

    def with_cache(self, use_cache: bool = True, ttl: int | None = None) -> "PipelineBuilder":
        """Enable or disable caching.

        Args:
            use_cache: Whether to use cache
            ttl: Optional cache TTL in seconds

        Returns:
            Self for method chaining
        """
        self._config.use_cache = use_cache
        if ttl is not None:
            self._config.cache_ttl = ttl
        return self

    def with_observer(self, observer: Observer) -> "PipelineBuilder":
        """Add an observer.

        Args:
            observer: Observer to add

        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self

    def build(self) -> OptimizationPipeline:
        """Build the pipeline.

        Returns:
            Configured OptimizationPipeline
        """
        pipeline = OptimizationPipeline(self._config)
        for observer in self._observers:
            pipeline.add_observer(observer)
        return pipeline