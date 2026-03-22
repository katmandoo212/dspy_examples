"""Tests for MetricObserver."""

import pytest
from datetime import datetime


class TestMetricObserver:
    """Tests for MetricObserver."""

    def test_metric_observer_creation(self):
        """Test creating a metric observer."""
        from dspy_examples.observers.metric_observer import MetricObserver

        observer = MetricObserver()
        assert observer is not None
        assert observer.metrics == {}

    def test_metric_observer_collects_metrics(self):
        """Test that metrics are collected."""
        from dspy_examples.observers.metric_observer import MetricObserver
        from dspy_examples.observers.base import MetricEvent

        observer = MetricObserver()

        # Add some metric events
        event1 = MetricEvent(
            name="tokens_used",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=100,
            unit="tokens",
            aggregation="sum",
        )
        event2 = MetricEvent(
            name="tokens_used",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=50,
            unit="tokens",
            aggregation="sum",
        )

        observer.on_metric_event(event1)
        observer.on_metric_event(event2)

        assert observer.metrics["tokens_used"] == 150

    def test_metric_observer_max_aggregation(self):
        """Test max aggregation."""
        from dspy_examples.observers.metric_observer import MetricObserver
        from dspy_examples.observers.base import MetricEvent

        observer = MetricObserver()

        event1 = MetricEvent(
            name="max_duration",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value=100,
            unit="ms",
            aggregation="max",
        )
        event2 = MetricEvent(
            name="max_duration",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value=200,
            unit="ms",
            aggregation="max",
        )

        observer.on_metric_event(event1)
        observer.on_metric_event(event2)

        assert observer.metrics["max_duration"] == 200

    def test_metric_observer_last_aggregation(self):
        """Test last aggregation."""
        from dspy_examples.observers.metric_observer import MetricObserver
        from dspy_examples.observers.base import MetricEvent

        observer = MetricObserver()

        event1 = MetricEvent(
            name="current_status",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value="running",
            unit="status",
            aggregation="last",
        )
        event2 = MetricEvent(
            name="current_status",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value="completed",
            unit="status",
            aggregation="last",
        )

        observer.on_metric_event(event1)
        observer.on_metric_event(event2)

        assert observer.metrics["current_status"] == "completed"

    def test_metric_observer_get_summary(self):
        """Test getting metric summary."""
        from dspy_examples.observers.metric_observer import MetricObserver
        from dspy_examples.observers.base import MetricEvent

        observer = MetricObserver()

        event1 = MetricEvent(
            name="tokens_used",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=100,
            unit="tokens",
        )
        event2 = MetricEvent(
            name="duration_ms",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            value=1500,
            unit="ms",
        )

        observer.on_metric_event(event1)
        observer.on_metric_event(event2)

        summary = observer.get_summary()

        assert summary["tokens_used"] == {"value": 100, "unit": "tokens"}
        assert summary["duration_ms"] == {"value": 1500, "unit": "ms"}