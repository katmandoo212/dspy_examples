"""Logging observer for event output.

Logs events to console or file using Python's logging module.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import IO

from dspy_examples.observers.base import Event, Observer, PipelineEvent, MetricEvent


class LoggingObserver(Observer):
    """Observer that logs events using Python's logging.

    Attributes:
        logger: Logger instance
        level: Log level for events
        format: Format string for log messages
    """

    def __init__(
        self,
        name: str = "dspy_examples",
        level: int = logging.INFO,
        handler: logging.Handler | None = None,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ) -> None:
        """Initialize logging observer.

        Args:
            name: Logger name
            level: Log level (default: INFO)
            handler: Optional custom handler (default: StreamHandler to stderr)
            format: Format string for log messages
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.level = level

        # Use provided handler or create default
        if handler:
            # Clear existing handlers and use the provided one
            self.logger.handlers.clear()
            self.logger.addHandler(handler)
        elif not self.logger.handlers:
            # No handlers exist and none provided, create default
            self.logger.addHandler(logging.StreamHandler())

        # Apply formatter to all handlers
        formatter = logging.Formatter(format)
        for h in self.logger.handlers:
            h.setFormatter(formatter)

    def on_event(self, event: Event) -> None:
        """Log any event."""
        message = f"[{event.source}] {event.name}"
        if event.data:
            message += f" | {event.data}"
        self.logger.log(self.level, message)

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Log pipeline event with stage and status."""
        message = f"[{event.source}] {event.stage}.{event.status}"
        if event.duration_ms:
            message += f" | {event.duration_ms}ms"
        if event.error:
            message += f" | ERROR: {event.error}"
        self.logger.log(self.level, message)

    def on_metric_event(self, event: MetricEvent) -> None:
        """Log metric event with value and unit."""
        message = f"[{event.source}] {event.name}: {event.value} {event.unit}"
        self.logger.log(self.level, message)