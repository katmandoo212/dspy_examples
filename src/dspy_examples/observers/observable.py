"""Observable mixin for classes that emit events.

Provides methods to manage observers and emit events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy_examples.observers.base import Observer, Event, PipelineEvent, MetricEvent


class Observable:
    """Mixin for classes that emit events.

    Classes using this mixin can add observers and emit events.
    """

    def __init__(self) -> None:
        """Initialize observable with empty observer list."""
        self._observers: list["Observer"] = []

    def add_observer(self, observer: "Observer") -> "Observable":
        """Add an observer.

        Args:
            observer: Observer to add

        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self

    def remove_observer(self, observer: "Observer") -> None:
        """Remove an observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _emit(self, event: "Event") -> None:
        """Emit an event to all observers.

        Args:
            event: Event to emit
        """
        from dspy_examples.observers.base import PipelineEvent, MetricEvent

        for observer in self._observers:
            observer.on_event(event)
            if isinstance(event, PipelineEvent):
                observer.on_pipeline_event(event)
            elif isinstance(event, MetricEvent):
                observer.on_metric_event(event)