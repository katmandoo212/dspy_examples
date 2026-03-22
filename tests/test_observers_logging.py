"""Tests for LoggingObserver."""

import pytest
from datetime import datetime
from io import StringIO
import logging


class TestLoggingObserver:
    """Tests for LoggingObserver."""

    def test_logging_observer_creation(self):
        """Test creating a logging observer."""
        from dspy_examples.observers.logging_observer import LoggingObserver

        observer = LoggingObserver()
        assert observer is not None

    def test_logging_observer_logs_events(self):
        """Test that events are logged."""
        from dspy_examples.observers.logging_observer import LoggingObserver
        from dspy_examples.observers.base import Event

        # Create string buffer to capture logs
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.INFO)

        observer = LoggingObserver(handler=handler)

        event = Event(
            name="test_event",
            timestamp=datetime.now(),
            source="test",
            data={"key": "value"},
        )
        observer.on_event(event)

        log_output = log_buffer.getvalue()
        assert "test_event" in log_output

    def test_logging_observer_pipeline_events(self):
        """Test logging pipeline events."""
        from dspy_examples.observers.logging_observer import LoggingObserver
        from dspy_examples.observers.base import PipelineEvent

        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.INFO)

        observer = LoggingObserver(handler=handler)

        event = PipelineEvent(
            name="pipeline_started",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="started",
        )
        observer.on_pipeline_event(event)

        log_output = log_buffer.getvalue()
        assert "optimize" in log_output
        assert "started" in log_output

    def test_logging_observer_with_level(self):
        """Test logging with different log levels."""
        from dspy_examples.observers.logging_observer import LoggingObserver
        from dspy_examples.observers.base import Event

        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)

        observer = LoggingObserver(level=logging.DEBUG, handler=handler)

        event = Event(
            name="debug_event",
            timestamp=datetime.now(),
            source="test",
            data={},
        )
        observer.on_event(event)

        log_output = log_buffer.getvalue()
        assert "debug_event" in log_output