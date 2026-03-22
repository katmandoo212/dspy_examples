"""Tests for CallbackObserver."""

import pytest
from datetime import datetime


class TestCallbackObserver:
    """Tests for CallbackObserver."""

    def test_callback_observer_with_function(self):
        """Test creating observer with callback function."""
        from dspy_examples.observers.callback_observer import CallbackObserver
        from dspy_examples.observers.base import Event

        events = []

        def capture(event):
            events.append(event)

        observer = CallbackObserver(callback=capture)

        event = Event(
            name="test",
            timestamp=datetime.now(),
            source="test",
            data={},
        )
        observer.on_event(event)

        assert len(events) == 1
        assert events[0].name == "test"

    def test_callback_observer_with_lambda(self):
        """Test creating observer with lambda."""
        from dspy_examples.observers.callback_observer import CallbackObserver
        from dspy_examples.observers.base import Event

        results = []
        observer = CallbackObserver(callback=lambda e: results.append(e.name))

        event = Event(
            name="lambda_test",
            timestamp=datetime.now(),
            source="test",
            data={},
        )
        observer.on_event(event)

        assert results == ["lambda_test"]

    def test_callback_observer_pipeline_handler(self):
        """Test callback for pipeline events."""
        from dspy_examples.observers.callback_observer import CallbackObserver
        from dspy_examples.observers.base import PipelineEvent

        stages = []

        def capture_stage(event):
            stages.append(event.stage)

        observer = CallbackObserver(
            on_pipeline=lambda e: stages.append(e.stage)
        )

        event = PipelineEvent(
            name="stage",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="started",
        )
        observer.on_pipeline_event(event)

        assert stages == ["optimize"]

    def test_callback_observer_metric_handler(self):
        """Test callback for metric events."""
        from dspy_examples.observers.callback_observer import CallbackObserver
        from dspy_examples.observers.base import MetricEvent

        values = []

        observer = CallbackObserver(
            on_metric=lambda e: values.append((e.name, e.value))
        )

        event = MetricEvent(
            name="tokens",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=100,
            unit="count",
        )
        observer.on_metric_event(event)

        assert values == [("tokens", 100)]

    def test_callback_observer_all_handlers(self):
        """Test callback with all handler types."""
        from dspy_examples.observers.callback_observer import CallbackObserver
        from dspy_examples.observers.base import Event, PipelineEvent, MetricEvent

        events = []
        pipeline_events = []
        metric_events = []

        observer = CallbackObserver(
            callback=lambda e: events.append(e),
            on_pipeline=lambda e: pipeline_events.append(e),
            on_metric=lambda e: metric_events.append(e),
        )

        observer.on_event(Event(name="e1", timestamp=datetime.now(), source="test", data={}))
        observer.on_pipeline_event(PipelineEvent(name="p1", timestamp=datetime.now(), source="test", data={}, stage="test", status="started"))
        observer.on_metric_event(MetricEvent(name="m1", timestamp=datetime.now(), source="test", data={}, value=1, unit="count"))

        assert len(events) == 1
        assert len(pipeline_events) == 1
        assert len(metric_events) == 1