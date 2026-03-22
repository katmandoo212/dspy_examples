"""BatchBuilder for fluent configuration.

Usage:
    batch = (BatchBuilder()
        .with_prompts(["p1.md", "p2.md"])
        .with_providers(["openai", "ollama"])
        .with_optimizer("gepa")
        .build())
"""

from __future__ import annotations

from pathlib import Path

from dspy_examples.commands import BatchCommand, BatchConfig
from dspy_examples.observers import Observer


class BatchBuilder:
    """Builder for BatchCommand with fluent API.

    Provides method chaining for batch configuration.

    Example:
        batch = (BatchBuilder()
            .with_prompts(["prompt1.md", "prompt2.md"])
            .with_providers(["openai", "ollama"])
            .with_optimizer("mipro_v2")
            .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._config = BatchConfig()
        self._observers: list[Observer] = []

    def with_prompts(self, paths: list[str | Path]) -> "BatchBuilder":
        """Set prompt paths.

        Args:
            paths: List of prompt file paths

        Returns:
            Self for method chaining
        """
        self._config.prompt_paths = [Path(p) for p in paths]
        return self

    def with_template(self, path: str | Path) -> "BatchBuilder":
        """Set template path for variable substitution.

        Args:
            path: Path to template file

        Returns:
            Self for method chaining
        """
        self._config.prompt_template = Path(path)
        return self

    def with_variables(self, variable_sets: list[dict[str, str]]) -> "BatchBuilder":
        """Set variable sets for template.

        Args:
            variable_sets: List of variable dictionaries

        Returns:
            Self for method chaining
        """
        self._config.variable_sets = variable_sets
        return self

    def with_providers(self, names: list[str]) -> "BatchBuilder":
        """Set provider names.

        Args:
            names: List of provider names

        Returns:
            Self for method chaining
        """
        self._config.providers = [{"name": name} for name in names]
        return self

    def with_provider(self, name: str, model: str | None = None) -> "BatchBuilder":
        """Add a single provider.

        Args:
            name: Provider name
            model: Optional model name (currently unused)

        Returns:
            Self for method chaining
        """
        provider = {"name": name}
        if model:
            provider["model"] = model
        self._config.providers.append(provider)
        return self

    def with_optimizer(self, name: str) -> "BatchBuilder":
        """Set optimizer.

        Args:
            name: Optimizer name

        Returns:
            Self for method chaining
        """
        self._config.optimizer_name = name
        return self

    def with_output_dir(self, path: str | Path) -> "BatchBuilder":
        """Set output directory.

        Args:
            path: Output directory path

        Returns:
            Self for method chaining
        """
        self._config.output_dir = Path(path)
        return self

    def with_observer(self, observer: Observer) -> "BatchBuilder":
        """Add an observer.

        Args:
            observer: Observer to add

        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self

    def build(self) -> BatchCommand:
        """Build the batch command.

        Returns:
            Configured BatchCommand
        """
        return BatchCommand(self._config)