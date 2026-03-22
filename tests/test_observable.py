"""Tests for Observable mixin."""

import pytest
from datetime import datetime


class TestObservable:
    """Tests for Observable mixin."""

    def test_observable_add_observer(self):
        """Test adding an observer."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer

        class MyObservable(Observable):
            pass

        class MyObserver(Observer):
            pass

        observable = MyObservable()
        observer = MyObserver()
        observable.add_observer(observer)

        assert observer in observable._observers

    def test_observable_remove_observer(self):
        """Test removing an observer."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer

        class MyObservable(Observable):
            pass

        class MyObserver(Observer):
            pass

        observable = MyObservable()
        observer = MyObserver()
        observable.add_observer(observer)
        observable.remove_observer(observer)

        assert observer not in observable._observers

    def test_observable_emit_event(self):
        """Test emitting events to observers."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer, Event

        class MyObservable(Observable):
            pass

        events = []

        class MyObserver(Observer):
            def on_event(self, event):
                events.append(event)

        observable = MyObservable()
        observable.add_observer(MyObserver())

        event = Event(name="test", timestamp=datetime.now(), source="test", data={})
        observable._emit(event)

        assert len(events) == 1
        assert events[0].name == "test"

    def test_observable_emit_pipeline_event(self):
        """Test emitting pipeline events."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer, PipelineEvent

        class MyObservable(Observable):
            pass

        events = []

        class MyObserver(Observer):
            def on_pipeline_event(self, event):
                events.append(event)

        observable = MyObservable()
        observable.add_observer(MyObserver())

        event = PipelineEvent(
            name="test",
            timestamp=datetime.now(),
            source="pipeline",
            data={},
            stage="optimize",
            status="started",
        )
        observable._emit(event)

        assert len(events) == 1

    def test_observable_emit_metric_event(self):
        """Test emitting metric events."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer, MetricEvent

        class MyObservable(Observable):
            pass

        events = []

        class MyObserver(Observer):
            def on_metric_event(self, event):
                events.append(event)

        observable = MyObservable()
        observable.add_observer(MyObserver())

        event = MetricEvent(
            name="tokens",
            timestamp=datetime.now(),
            source="optimizer",
            data={},
            value=100,
            unit="count",
        )
        observable._emit(event)

        assert len(events) == 1

    def test_observable_method_chaining(self):
        """Test that add_observer returns self for chaining."""
        from dspy_examples.observers.observable import Observable
        from dspy_examples.observers.base import Observer

        class MyObservable(Observable):
            pass

        class MyObserver(Observer):
            pass

        observable = MyObservable()
        result = observable.add_observer(MyObserver())

        assert result is observable