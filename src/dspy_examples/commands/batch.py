"""High-level BatchCommand API for batch processing.

Usage:
    from dspy_examples.commands import BatchCommand, BatchConfig
    from pathlib import Path

    # Multiple prompts, single provider
    config = BatchConfig(
        prompt_paths=[Path("prompts/q1.md"), Path("prompts/q2.md")],
        providers=[{"name": "openai", "model": "gpt-4"}],
    )
    batch = BatchCommand(config)
    result = batch.run()

    # Multi-provider comparison
    config = BatchConfig(
        prompt_paths=[Path("prompts/test.md")],
        providers=[
            {"name": "openai", "model": "gpt-4"},
            {"name": "anthropic", "model": "claude-3"},
        ],
    )
    result = batch.run()

    # Template with variable sets
    config = BatchConfig(
        prompt_template=Path("prompts/template.md"),
        variable_sets=[
            {"country": "France", "tone": "formal"},
            {"country": "Japan", "tone": "casual"},
        ],
        providers=[{"name": "ollama"}],
    )
    result = batch.run()
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from dspy_examples.commands.base import Command, CommandResult
from dspy_examples.commands.flows import BatchConfig, BatchFlow
from dspy_examples.commands.queue import CommandQueue
from dspy_examples.commands.results import ResultsAggregator


class BatchCommand:
    """High-level API for batch prompt optimization.

    Orchestrates the full batch processing workflow:
    1. Generate configuration combinations
    2. Create queue for persistence
    3. Execute each configuration
    4. Aggregate results
    5. Generate reports
    """

    def __init__(self, config: BatchConfig) -> None:
        """Initialize batch command with configuration.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self._flow = BatchFlow(config)
        self.batch_id = self._flow.get_batch_id()
        self._queue_path = (
            self.config.output_dir / ".cache" / f"{self.batch_id}_queue.db"
        )

    def get_configs(self) -> list[dict[str, Any]]:
        """Get all configuration combinations.

        Returns:
            List of configuration dicts for each combination.
        """
        return self._flow.get_configs()

    def run(
        self,
        queue_path: Path | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run the batch processing.

        Args:
            queue_path: Optional path for queue database
            resume: Whether to resume from previous queue

        Returns:
            Dictionary with batch results:
            - batch_id: Unique identifier
            - total_commands: Total number of commands
            - successful: Number of successful commands
            - failed: Number of failed commands
            - total_time: Total execution time
            - results: List of individual results
            - report_path: Path to Markdown report
        """
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        # Setup queue for persistence
        if queue_path:
            self._queue_path = queue_path

        queue = CommandQueue(self._queue_path)

        # Get configurations
        configs = self.get_configs()

        # Add commands to queue if not resuming
        if not resume or queue.is_empty():
            queue.clear()
            for i, cfg in enumerate(configs):
                cmd = OptimizeCommand(
                    command_id=f"{self.batch_id}_{i:03d}",
                    prompt_path=cfg["prompt_path"],
                    output_path=cfg["output_path"],
                    provider_name=cfg["provider_name"],
                    model_name=cfg.get("model_name"),
                    optimizer_name=cfg["optimizer_name"],
                    variables=cfg.get("variables", {}),
                )
                queue.add(cmd, batch_id=self.batch_id)

        # Setup aggregator
        aggregator = ResultsAggregator()

        # Process each configuration
        start_time = time.time()

        # Get pending configs
        pending = queue.get_pending()

        for cmd_id, cmd_data in pending:
            try:
                # Run optimization
                result = self._run_optimization(cmd_data)

                # Save result
                queue.save_result(cmd_id, {
                    "status": "success",
                    "output_path": str(result.output_path),
                    "optimizer_name": result.optimizer_name,
                    "provider_name": result.provider_name,
                    "model_name": result.model_name or "default",
                    "execution_time": result.execution_time,
                    "metadata": {},
                })

                # Mark completed
                queue.mark_completed(cmd_id)

                # Add to aggregator
                aggregator.add(CommandResult(
                    command_id=cmd_id,
                    status="success",
                    output_path=result.output_path,
                    optimizer_name=result.optimizer_name,
                    provider_name=result.provider_name,
                    model_name=result.model_name or "default",
                    execution_time=result.execution_time,
                ))

            except Exception as e:
                # Mark failed
                queue.mark_failed(cmd_id, str(e))

                # Add failed result
                aggregator.add(CommandResult(
                    command_id=cmd_id,
                    status="failed",
                    output_path=None,
                    optimizer_name=cmd_data.get("optimizer_name", "unknown"),
                    provider_name=cmd_data.get("provider_name", "unknown"),
                    model_name=cmd_data.get("model_name", "unknown"),
                    execution_time=0.0,
                    error_message=str(e),
                ))

        total_time = time.time() - start_time

        # Generate batch result
        batch_result = aggregator.aggregate()
        batch_result.total_time = total_time

        # Save reports
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        batch_result.save(self.config.output_dir)

        return {
            "batch_id": self.batch_id,
            "total_commands": batch_result.total_commands,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
            "total_time": batch_result.total_time,
            "results": batch_result.results,
            "report_path": self.config.output_dir / f"{self.batch_id}_report.md",
        }

    def _run_optimization(
        self, config: dict[str, Any]
    ) -> CommandResult:
        """Run single optimization from config.

        Args:
            config: Configuration dict

        Returns:
            CommandResult
        """
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        pipeline_config = PipelineConfig(
            provider_name=config["provider_name"],
            optimizer_name=config["optimizer_name"],
            input_path=Path(config["prompt_path"]),
            output_path=Path(config["output_path"]),
            variables=config.get("variables", {}),
            use_cache=False,  # Don't use cache in batch
        )

        pipeline = OptimizationPipeline(pipeline_config)
        result = pipeline.run(config.get("variables", {}))

        return CommandResult(
            command_id="",  # Set by caller
            status="success",
            output_path=Path(config["output_path"]),
            optimizer_name=result.optimizer_name,
            provider_name=result.provider_name,
            model_name=config.get("model_name"),
            execution_time=result.execution_time or 0.0,
        )


class OptimizeCommand(Command):
    """Command for single prompt optimization.

    Implements Command interface for queue persistence.
    """

    def __init__(
        self,
        command_id: str,
        prompt_path: Path,
        output_path: Path,
        provider_name: str,
        model_name: str | None,
        optimizer_name: str,
        variables: dict[str, str],
    ) -> None:
        self._id = command_id
        self.prompt_path = Path(prompt_path)
        self.output_path = Path(output_path)
        self.provider_name = provider_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.variables = variables

    @property
    def command_id(self) -> str:
        return self._id

    def execute(self) -> CommandResult:
        """Execute the optimization."""
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        config = PipelineConfig(
            provider_name=self.provider_name,
            optimizer_name=self.optimizer_name,
            input_path=self.prompt_path,
            output_path=self.output_path,
            variables=self.variables,
            use_cache=False,
        )

        pipeline = OptimizationPipeline(config)
        result = pipeline.run(self.variables)

        return CommandResult(
            command_id=self._id,
            status="success",
            output_path=self.output_path,
            optimizer_name=result.optimizer_name,
            provider_name=result.provider_name,
            model_name=self.model_name or "default",
            execution_time=result.execution_time or 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self._id,
            "prompt_path": str(self.prompt_path),
            "output_path": str(self.output_path),
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "optimizer_name": self.optimizer_name,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeCommand":
        """Deserialize from dictionary."""
        return cls(
            command_id=data["id"],
            prompt_path=Path(data["prompt_path"]),
            output_path=Path(data["output_path"]),
            provider_name=data["provider_name"],
            model_name=data.get("model_name"),
            optimizer_name=data["optimizer_name"],
            variables=data.get("variables", {}),
        )