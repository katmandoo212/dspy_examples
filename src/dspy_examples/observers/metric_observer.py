"""Metric observer for collecting performance data.

Aggregates metrics by name using sum, max, or last aggregation.
"""

from __future__ import annotations

from typing import Any

from dspy_examples.observers.base import Observer, MetricEvent


class MetricObserver(Observer):
    """Observer that collects and aggregates metrics.

    Attributes:
        metrics: Dictionary of metric name to aggregated value
        units: Dictionary of metric name to unit
    """

    def __init__(self) -> None:
        """Initialize metric observer."""
        self.metrics: dict[str, float | str] = {}
        self.units: dict[str, str] = {}

    def on_metric_event(self, event: MetricEvent) -> None:
        """Aggregate metric value."""
        name = event.name

        # Store unit
        self.units[name] = event.unit

        # Aggregate value
        if event.aggregation == "sum":
            current = self.metrics.get(name, 0)
            self.metrics[name] = current + event.value
        elif event.aggregation == "max":
            current = self.metrics.get(name, 0)
            self.metrics[name] = max(current, event.value)
        elif event.aggregation == "last":
            self.metrics[name] = event.value

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all metrics.

        Returns:
            Dictionary mapping metric name to value and unit.
        """
        return {
            name: {"value": value, "unit": self.units.get(name, "")}
            for name, value in self.metrics.items()
        }

    def reset(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.units.clear()