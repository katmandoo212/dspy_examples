"""Batch processing flows using PocketFlow.

Provides high-level batch processing with:
- Multiple prompts, single config
- Single prompt, multiple variable sets
- Multi-provider comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid

from dspy_examples.pocketflow import Node


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        prompt_paths: List of prompt files to process
        prompt_template: Template file for variable substitution
        variable_sets: List of variable dictionaries for substitution
        providers: List of provider configurations
        optimizer_name: Name of optimizer to use
        optimizer_config: Optional optimizer configuration
        output_dir: Directory for output files
        naming_pattern: Pattern for output file names
        max_concurrent: Maximum concurrent executions
        retry_count: Number of retries for failed commands
    """

    # Prompt sources
    prompt_paths: list[Path] | None = None
    prompt_template: Path | None = None
    variable_sets: list[dict[str, str]] | None = None

    # Provider configurations
    providers: list[dict[str, str]] = field(default_factory=lambda: [{"name": "ollama"}])

    # Optimizer settings
    optimizer_name: str = "bootstrap_fewshot"
    optimizer_config: dict[str, Any] | None = None

    # Output settings
    output_dir: Path = Path("prompts/batch_output")
    naming_pattern: str = "{prompt}_{provider}_{model}_{optimizer}"

    # Execution settings
    max_concurrent: int = 1
    retry_count: int = 2


class BatchFlow:
    """Orchestrates batch optimization processing.

    Creates and manages batches of OptimizeNodes.
    """

    def __init__(self, config: BatchConfig) -> None:
        """Initialize batch flow with configuration.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.batch_id = self._generate_batch_id()
        self._build_flow()

    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier."""
        return f"batch_{uuid.uuid4().hex[:8]}"

    def _build_flow(self) -> None:
        """Build the flow of OptimizeNodes."""
        configs = self._generate_configs()
        self._configs = configs

    def _generate_configs(self) -> list[dict[str, Any]]:
        """Generate all configuration combinations.

        Returns:
            List of config dicts for each combination of:
            - Prompt × Provider × Variables
        """
        configs = []

        # Get prompt sources
        if self.config.prompt_paths:
            prompt_sources = [(p, None) for p in self.config.prompt_paths]
        elif self.config.prompt_template and self.config.variable_sets:
            prompt_sources = [
                (self.config.prompt_template, vars)
                for vars in self.config.variable_sets
            ]
        elif self.config.prompt_template:
            prompt_sources = [(self.config.prompt_template, None)]
        else:
            raise ValueError("Must provide prompt_paths or prompt_template")

        # Generate all combinations
        for prompt_path, variables in prompt_sources:
            for provider_config in self.config.providers:
                provider_name = provider_config.get("name", "ollama")
                model_name = provider_config.get("model")

                config = {
                    "prompt_path": prompt_path,
                    "variables": variables or {},
                    "provider_name": provider_name,
                    "model_name": model_name,
                    "optimizer_name": self.config.optimizer_name,
                    "output_path": self._get_output_path(
                        prompt_path, provider_name, model_name
                    ),
                }
                configs.append(config)

        return configs

    def _get_output_path(
        self, prompt_path: Path, provider_name: str, model_name: str | None
    ) -> Path:
        """Generate output file path for a configuration."""
        prompt_stem = prompt_path.stem
        model = model_name or "default"
        optimizer = self.config.optimizer_name

        filename = self.config.naming_pattern.format(
            prompt=prompt_stem,
            provider=provider_name,
            model=model,
            optimizer=optimizer,
        )

        return self.config.output_dir / f"{filename}.md"

    def get_configs(self) -> list[dict[str, Any]]:
        """Get all configuration combinations."""
        return self._configs

    def get_batch_id(self) -> str:
        """Get the batch identifier."""
        return self.batch_id

    def create_nodes(self) -> list[Node]:
        """Create OptimizeNodes for all configurations."""
        from dspy_examples.commands.nodes import OptimizeNode

        nodes = []
        for config in self._configs:
            node = OptimizeNode(
                prompt_path=config["prompt_path"],
                output_path=config["output_path"],
                provider_name=config["provider_name"],
                model_name=config["model_name"],
                optimizer_name=config["optimizer_name"],
                variables=config["variables"],
            )
            nodes.append(node)
        return nodes