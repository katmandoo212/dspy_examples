"""Callback observer for custom event handling.

Wraps user-provided callback functions.
"""

from __future__ import annotations

from typing import Callable

from dspy_examples.observers.base import Observer, Event, PipelineEvent, MetricEvent


class CallbackObserver(Observer):
    """Observer that calls user-provided callback functions.

    Attributes:
        callback: Function called for any event
        on_pipeline: Function called for pipeline events
        on_metric: Function called for metric events
    """

    def __init__(
        self,
        callback: Callable[[Event], None] | None = None,
        on_pipeline: Callable[[PipelineEvent], None] | None = None,
        on_metric: Callable[[MetricEvent], None] | None = None,
    ) -> None:
        """Initialize callback observer.

        Args:
            callback: Function called for any event
            on_pipeline: Function called for pipeline events
            on_metric: Function called for metric events
        """
        self.callback = callback
        self.on_pipeline = on_pipeline
        self.on_metric = on_metric

    def on_event(self, event: Event) -> None:
        """Call callback for any event."""
        if self.callback:
            self.callback(event)

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Call pipeline callback."""
        if self.on_pipeline:
            self.on_pipeline(event)

    def on_metric_event(self, event: MetricEvent) -> None:
        """Call metric callback."""
        if self.on_metric:
            self.on_metric(event)