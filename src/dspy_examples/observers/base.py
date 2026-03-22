"""Base classes for observer pattern.

Provides event system for tracking optimization progress,
collecting metrics, and enabling extensibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """Base event class for all observer events.

    Attributes:
        name: Event name (e.g., "optimization_started")
        timestamp: When the event occurred
        source: Source component (e.g., "pipeline", "optimizer")
        data: Event-specific data
        metadata: User-extensible metadata
    """

    name: str
    timestamp: datetime
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEvent(Event):
    """Event for pipeline lifecycle stages.

    Attributes:
        stage: Pipeline stage ("load_prompt", "configure", "optimize", "save")
        status: Stage status ("started", "completed", "failed")
        duration_ms: Duration in milliseconds (optional)
        error: Error message if failed (optional)
    """

    stage: str = ""
    status: str = ""
    duration_ms: int | None = None
    error: str | None = None


@dataclass
class MetricEvent(Event):
    """Event for performance metrics.

    Attributes:
        value: The metric value
        unit: Unit of measurement ("tokens", "seconds", "attempts")
        aggregation: How to aggregate ("sum", "max", "last")
    """

    value: float = 0.0
    unit: str = ""
    aggregation: str = "sum"


class Observer:
    """Abstract base class for observers.

    Observers receive events and can react to them.
    Subclasses can override specific event handlers.
    """

    def __init__(self) -> None:
        """Prevent direct instantiation of Observer."""
        if self.__class__ is Observer:
            raise TypeError("Cannot instantiate abstract class Observer")

    def on_event(self, event: Event) -> None:
        """Called when any event is emitted.

        Override this to handle all events uniformly.

        Args:
            event: The event that was emitted
        """
        pass

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Called for pipeline lifecycle events.

        Args:
            event: The pipeline event
        """
        pass

    def on_metric_event(self, event: MetricEvent) -> None:
        """Called for metric events.

        Args:
            event: The metric event
        """
        pass