"""Result aggregation and reporting for batch processing.

Generates:
- Individual output files for each optimization
- Summary report in Markdown format
- Full results in JSON format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dspy_examples.commands.base import CommandResult


@dataclass
class BatchResult:
    """Result of a batch optimization run.

    Attributes:
        batch_id: Unique identifier for this batch
        total_commands: Total number of commands in batch
        successful: Number of successful commands
        failed: Number of failed commands
        skipped: Number of skipped commands
        total_time: Total execution time in seconds
        results: List of individual command results
        by_provider: Statistics grouped by provider
        by_optimizer: Statistics grouped by optimizer
    """

    batch_id: str
    total_commands: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    results: list[CommandResult]
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_optimizer: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert batch result to Markdown report."""
        lines = [
            f"# Batch Report: {self.batch_id}",
            "",
            "## Summary",
            f"- Total: {self.total_commands} commands",
            f"- Successful: {self.successful}",
            f"- Failed: {self.failed}",
            f"- Total time: {self.total_time:.1f}s",
            "",
        ]

        # Provider statistics
        if self.by_provider:
            lines.append("## By Provider")
            lines.append("| Provider | Model | Success | Failed | Avg Time |")
            lines.append("|----------|-------|---------|--------|----------|")

            for provider, stats in self.by_provider.items():
                model = stats.get("model", "default")
                success = stats.get("success", 0)
                failed = stats.get("failed", 0)
                avg_time = stats.get("avg_time", 0.0)
                lines.append(
                    f"| {provider} | {model} | {success} | {failed} | {avg_time:.1f}s |"
                )

            lines.append("")

        # Optimizer statistics
        if self.by_optimizer:
            lines.append("## By Optimizer")
            lines.append("| Optimizer | Avg Time | Success Rate |")
            lines.append("|-----------|----------|--------------|")

            for optimizer, stats in self.by_optimizer.items():
                avg_time = stats.get("avg_time", 0.0)
                success_rate = stats.get("success_rate", 0.0)
                lines.append(
                    f"| {optimizer} | {avg_time:.1f}s | {success_rate:.0%} |"
                )

            lines.append("")

        # Failed commands
        failed_results = [r for r in self.results if r.status == "failed"]
        if failed_results:
            lines.append("## Failed Commands")
            for result in failed_results:
                lines.append(
                    f"- {result.command_id}: {result.error_message or 'Unknown error'}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert batch result to JSON dict."""
        return {
            "batch_id": self.batch_id,
            "total_commands": self.total_commands,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_time": self.total_time,
            "results": [
                {
                    "command_id": r.command_id,
                    "status": r.status,
                    "output_path": str(r.output_path) if r.output_path else None,
                    "optimizer_name": r.optimizer_name,
                    "provider_name": r.provider_name,
                    "model_name": r.model_name,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "by_provider": self.by_provider,
            "by_optimizer": self.by_optimizer,
        }

    def save(self, output_dir: Path) -> None:
        """Save batch result to files.

        Creates:
            - {batch_id}_report.md: Markdown summary
            - {batch_id}_results.json: Full JSON results

        Args:
            output_dir: Directory to save files in
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown report
        md_path = output_dir / f"{self.batch_id}_report.md"
        md_path.write_text(self.to_markdown())

        # Save JSON results
        json_path = output_dir / f"{self.batch_id}_results.json"
        json_path.write_text(json.dumps(self.to_json(), indent=2))


class ResultsAggregator:
    """Aggregates results from batch processing.

    Collects CommandResults and computes statistics.
    """

    def __init__(self) -> None:
        """Initialize empty aggregator."""
        self.results: list[CommandResult] = []

    def add(self, result: CommandResult) -> None:
        """Add a command result.

        Args:
            result: Command result to add
        """
        self.results.append(result)

    def aggregate(self) -> BatchResult:
        """Create BatchResult with statistics.

        Returns:
            BatchResult with aggregated statistics
        """
        if not self.results:
            return BatchResult(
                batch_id="empty",
                total_commands=0,
                successful=0,
                failed=0,
                skipped=0,
                total_time=0.0,
                results=[],
            )

        # Generate batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Count by status
        successful = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        total_time = sum(r.execution_time for r in self.results)

        # Compute by-provider statistics
        by_provider: dict[str, dict[str, Any]] = {}
        for r in self.results:
            key = r.provider_name
            if key not in by_provider:
                by_provider[key] = {
                    "model": r.model_name,
                    "count": 0,
                    "success": 0,
                    "failed": 0,
                    "total_time": 0.0,
                }
            by_provider[key]["count"] += 1
            by_provider[key]["total_time"] += r.execution_time
            if r.status == "success":
                by_provider[key]["success"] += 1
            elif r.status == "failed":
                by_provider[key]["failed"] += 1

        # Compute averages
        for key in by_provider:
            stats = by_provider[key]
            count = stats["count"]
            stats["avg_time"] = stats["total_time"] / count if count > 0 else 0.0
            stats["success_rate"] = stats["success"] / count if count > 0 else 0.0

        # Compute by-optimizer statistics
        by_optimizer: dict[str, dict[str, Any]] = {}
        for r in self.results:
            key = r.optimizer_name
            if key not in by_optimizer:
                by_optimizer[key] = {
                    "count": 0,
                    "success": 0,
                    "total_time": 0.0,
                }
            by_optimizer[key]["count"] += 1
            by_optimizer[key]["total_time"] += r.execution_time
            if r.status == "success":
                by_optimizer[key]["success"] += 1

        # Compute averages
        for key in by_optimizer:
            stats = by_optimizer[key]
            count = stats["count"]
            stats["avg_time"] = stats["total_time"] / count if count > 0 else 0.0
            stats["success_rate"] = stats["success"] / count if count > 0 else 0.0

        return BatchResult(
            batch_id=batch_id,
            total_commands=len(self.results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time=total_time,
            results=self.results,
            by_provider=by_provider,
            by_optimizer=by_optimizer,
        )