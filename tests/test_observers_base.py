"""Tests for observer base classes."""

import pytest
from datetime import datetime


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        from dspy_examples.observers.base import Event

        event = Event(
            name="test_event",
            timestamp=datetime.now(),
            source="test",
            data={"key": "value"},
        )

        assert event.name == "test_event"
        assert event.source == "test"
        assert event.data["key"] == "value"
        assert event.metadata == {}

    def test_event_with_metadata(self):
        """Test event with user metadata."""
        from dspy_examples.observers.base import Event

        event = Event(
            name="test_event",
            timestamp=datetime.now(),
            source="test",
            data={},
            metadata={"user_id": "123"},
        )

        assert event.metadata["user_id"] == "123"


class TestPipelineEvent:
    """Tests for PipelineEvent."""

    def test_pipeline_event_creation(self):
        """Test creating a pipeline event."""
        from dspy_examples.observers.base import PipelineEvent

        event = PipelineEvent(
            name="optimization_started",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="started",
        )

        assert event.stage == "optimize"
        assert event.status == "started"
        assert event.duration_ms is None

    def test_pipeline_event_with_duration(self):
        """Test pipeline event with duration."""
        from dspy_examples.observers.base import PipelineEvent

        event = PipelineEvent(
            name="optimization_completed",
            timestamp=datetime.now(),
            source="pipeline",
            data={"result_length": 500},
            stage="optimize",
            status="completed",
            duration_ms=1500,
        )

        assert event.duration_ms == 1500

    def test_pipeline_event_failed(self):
        """Test pipeline event for failure."""
        from dspy_examples.observers.base import PipelineEvent

        event = PipelineEvent(
            name="optimization_failed",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="failed",
            error="Connection timeout",
        )

        assert event.status == "failed"
        assert event.error == "Connection timeout"


class TestMetricEvent:
    """Tests for MetricEvent."""

    def test_metric_event_creation(self):
        """Test creating a metric event."""
        from dspy_examples.observers.base import MetricEvent

        event = MetricEvent(
            name="tokens_used",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=1500,
            unit="tokens",
        )

        assert event.name == "tokens_used"
        assert event.value == 1500
        assert event.unit == "tokens"
        assert event.aggregation == "sum"  # default

    def test_metric_event_aggregation(self):
        """Test metric event with different aggregation."""
        from dspy_examples.observers.base import MetricEvent

        event = MetricEvent(
            name="max_duration",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value=2500,
            unit="ms",
            aggregation="max",
        )

        assert event.aggregation == "max"


class TestObserver:
    """Tests for Observer abstract class."""

    def test_observer_cannot_instantiate(self):
        """Test that Observer cannot be instantiated directly."""
        from dspy_examples.observers.base import Observer

        with pytest.raises(TypeError):
            Observer()

    def test_observer_implementation(self):
        """Test implementing a concrete observer."""
        from dspy_examples.observers.base import Observer, Event

        class MyObserver(Observer):
            def __init__(self):
                self.events = []

            def on_event(self, event: Event) -> None:
                self.events.append(event)

        observer = MyObserver()
        event = Event(name="test", timestamp=datetime.now(), source="test", data={})
        observer.on_event(event)

        assert len(observer.events) == 1
        assert observer.events[0].name == "test"

    def test_observer_pipeline_handler(self):
        """Test observer pipeline event handler."""
        from dspy_examples.observers.base import Observer, PipelineEvent

        class PipelineTracker(Observer):
            def __init__(self):
                self.stages = []

            def on_pipeline_event(self, event: PipelineEvent) -> None:
                self.stages.append(event.stage)

        tracker = PipelineTracker()
        event = PipelineEvent(
            name="stage",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="started",
        )
        tracker.on_pipeline_event(event)

        assert tracker.stages == ["optimize"]