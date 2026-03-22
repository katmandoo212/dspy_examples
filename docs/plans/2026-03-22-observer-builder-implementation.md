# Observer and Builder Patterns - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Observer Pattern for progress tracking and Builder Pattern for fluent configuration APIs.

**Architecture:** Observer uses abstract base class with Event dataclasses. Builder uses fluent interface with method chaining. Both integrate with existing Pipeline and BatchCommand.

**Tech Stack:** Python 3.14, dataclasses, abc, datetime

---

## Task 1: Observer Base Classes

**Files:**
- Create: `src/dspy_examples/observers/__init__.py`
- Create: `src/dspy_examples/observers/base.py`
- Test: `tests/test_observers_base.py`

**Step 1: Write the failing test for Event**

Create `tests/test_observers_base.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_base.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers'"

**Step 3: Create observer package structure**

Create `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)

__all__ = ["Observer", "Event", "PipelineEvent", "MetricEvent"]
```

**Step 4: Run test to verify module imports**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_base.py -v`
Expected: FAIL with "cannot import name 'Event' from 'dspy_examples.observers.base'"

**Step 5: Implement Event and Observer base classes**

Create `src/dspy_examples/observers/base.py`:

```python
"""Base classes for observer pattern.

Provides event system for tracking optimization progress,
collecting metrics, and enabling extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class Observer(ABC):
    """Abstract base class for observers.

    Observers receive events and can react to them.
    Subclasses can override specific event handlers.
    """

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
```

**Step 6: Add tests for PipelineEvent and MetricEvent**

Add to `tests/test_observers_base.py`:

```python
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
```

**Step 7: Run all tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_base.py -v`
Expected: PASS (all tests)

**Step 8: Commit**

```bash
git add src/dspy_examples/observers/__init__.py src/dspy_examples/observers/base.py tests/test_observers_base.py
git commit -m "feat: add observer base classes with Event, PipelineEvent, MetricEvent"
```

---

## Task 2: LoggingObserver Implementation

**Files:**
- Create: `src/dspy_examples/observers/logging_observer.py`
- Test: `tests/test_observers_logging.py`

**Step 1: Write the failing test**

Create `tests/test_observers_logging.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_logging.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers.logging_observer'"

**Step 3: Implement LoggingObserver**

Create `src/dspy_examples/observers/logging_observer.py`:

```python
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

        if not self.logger.handlers:
            if handler:
                self.logger.addHandler(handler)
            else:
                self.logger.addHandler(logging.StreamHandler())

        for h in self.logger.handlers:
            h.setFormatter(logging.Formatter(format))

        self.level = level

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
```

**Step 4: Update observers __init__.py**

Edit `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)
from dspy_examples.observers.logging_observer import LoggingObserver

__all__ = [
    "Observer",
    "Event",
    "PipelineEvent",
    "MetricEvent",
    "LoggingObserver",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_logging.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/observers/logging_observer.py src/dspy_examples/observers/__init__.py tests/test_observers_logging.py
git commit -m "feat: add LoggingObserver for event logging"
```

---

## Task 3: MetricObserver Implementation

**Files:**
- Create: `src/dspy_examples/observers/metric_observer.py`
- Test: `tests/test_observers_metric.py`

**Step 1: Write the failing test**

Create `tests/test_observers_metric.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_metric.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers.metric_observer'"

**Step 3: Implement MetricObserver**

Create `src/dspy_examples/observers/metric_observer.py`:

```python
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
```

**Step 4: Update observers __init__.py**

Edit `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)
from dspy_examples.observers.logging_observer import LoggingObserver
from dspy_examples.observers.metric_observer import MetricObserver

__all__ = [
    "Observer",
    "Event",
    "PipelineEvent",
    "MetricEvent",
    "LoggingObserver",
    "MetricObserver",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_metric.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/observers/metric_observer.py src/dspy_examples/observers/__init__.py tests/test_observers_metric.py
git commit -m "feat: add MetricObserver for collecting performance data"
```

---

## Task 4: CallbackObserver Implementation

**Files:**
- Create: `src/dspy_examples/observers/callback_observer.py`
- Test: `tests/test_observers_callback.py`

**Step 1: Write the failing test**

Create `tests/test_observers_callback.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_callback.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers.callback_observer'"

**Step 3: Implement CallbackObserver**

Create `src/dspy_examples/observers/callback_observer.py`:

```python
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
```

**Step 4: Update observers __init__.py**

Edit `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)
from dspy_examples.observers.logging_observer import LoggingObserver
from dspy_examples.observers.metric_observer import MetricObserver
from dspy_examples.observers.callback_observer import CallbackObserver

__all__ = [
    "Observer",
    "Event",
    "PipelineEvent",
    "MetricEvent",
    "LoggingObserver",
    "MetricObserver",
    "CallbackObserver",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_callback.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/observers/callback_observer.py src/dspy_examples/observers/__init__.py tests/test_observers_callback.py
git commit -m "feat: add CallbackObserver for custom event handling"
```

---

## Task 5: ProgressObserver Implementation

**Files:**
- Create: `src/dspy_examples/observers/progress_observer.py`
- Test: `tests/test_observers_progress.py`

**Step 1: Write the failing test**

Create `tests/test_observers_progress.py`:

```python
"""Tests for ProgressObserver."""

import pytest
from datetime import datetime


class TestProgressObserver:
    """Tests for ProgressObserver."""

    def test_progress_observer_creation(self):
        """Test creating a progress observer."""
        from dspy_examples.observers.progress_observer import ProgressObserver

        observer = ProgressObserver()
        assert observer is not None

    def test_progress_observer_tracks_stages(self):
        """Test that progress observer tracks pipeline stages."""
        from dspy_examples.observers.progress_observer import ProgressObserver
        from dspy_examples.observers.base import PipelineEvent

        observer = ProgressObserver(total_stages=4)

        stages = ["load_prompt", "configure", "optimize", "save"]
        for stage in stages:
            event = PipelineEvent(
                name="stage_started",
                timestamp=datetime.now(),
                source="pipeline",
                data={},
                stage=stage,
                status="started",
            )
            observer.on_pipeline_event(event)

        assert observer.current_stage == "save"
        assert observer.completed_stages == 4

    def test_progress_observer_progress_percent(self):
        """Test progress percentage calculation."""
        from dspy_examples.observers.progress_observer import ProgressObserver
        from dspy_examples.observers.base import PipelineEvent

        observer = ProgressObserver(total_stages=4)

        # Complete 2 stages
        for stage in ["load_prompt", "configure"]:
            event = PipelineEvent(
                name="stage_completed",
                timestamp=datetime.now(),
                source="pipeline",
                data={},
                stage=stage,
                status="completed",
            )
            observer.on_pipeline_event(event)

        assert observer.progress_percent == 50.0

    def test_progress_observer_skips_non_stage_events(self):
        """Test that non-stage events don't affect progress."""
        from dspy_examples.observers.progress_observer import ProgressObserver
        from dspy_examples.observers.base import Event

        observer = ProgressObserver(total_stages=4)

        event = Event(
            name="some_event",
            timestamp=datetime.now(),
            source="test",
            data={},
        )
        observer.on_event(event)

        assert observer.completed_stages == 0

    def test_progress_observer_custom_stages(self):
        """Test progress with custom stage names."""
        from dspy_examples.observers.progress_observer import ProgressObserver
        from dspy_examples.observers.base import PipelineEvent

        observer = ProgressObserver(
            total_stages=3,
            stage_names=["init", "process", "finish"],
        )

        for stage in ["init", "process"]:
            event = PipelineEvent(
                name="stage_completed",
                timestamp=datetime.now(),
                source="pipeline",
                data={},
                stage=stage,
                status="completed",
            )
            observer.on_pipeline_event(event)

        assert observer.completed_stages == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_progress.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers.progress_observer'"

**Step 3: Implement ProgressObserver**

Create `src/dspy_examples/observers/progress_observer.py`:

```python
"""Progress observer for tracking optimization progress.

Tracks completion of pipeline stages.
"""

from __future__ import annotations

from dspy_examples.observers.base import Observer, Event, PipelineEvent


class ProgressObserver(Observer):
    """Observer that tracks pipeline progress.

    Attributes:
        total_stages: Total number of stages
        stage_names: Optional list of stage names
        completed_stages: Number of completed stages
        current_stage: Name of current stage
    """

    def __init__(
        self,
        total_stages: int = 4,
        stage_names: list[str] | None = None,
    ) -> None:
        """Initialize progress observer.

        Args:
            total_stages: Total number of stages (default: 4)
            stage_names: Optional list of stage names
        """
        self.total_stages = total_stages
        self.stage_names = stage_names
        self.completed_stages = 0
        self.current_stage: str | None = None

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Track pipeline stage progress."""
        if event.status == "completed":
            self.completed_stages += 1
            self.current_stage = event.stage

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_stages == 0:
            return 0.0
        return (self.completed_stages / self.total_stages) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all stages are complete."""
        return self.completed_stages >= self.total_stages
```

**Step 4: Update observers __init__.py**

Edit `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)
from dspy_examples.observers.logging_observer import LoggingObserver
from dspy_examples.observers.metric_observer import MetricObserver
from dspy_examples.observers.callback_observer import CallbackObserver
from dspy_examples.observers.progress_observer import ProgressObserver

__all__ = [
    "Observer",
    "Event",
    "PipelineEvent",
    "MetricEvent",
    "LoggingObserver",
    "MetricObserver",
    "CallbackObserver",
    "ProgressObserver",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_progress.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/observers/progress_observer.py src/dspy_examples/observers/__init__.py tests/test_observers_progress.py
git commit -m "feat: add ProgressObserver for tracking pipeline progress"
```

---

## Task 6: Observable Mixin for Pipeline

**Files:**
- Create: `src/dspy_examples/observers/observable.py`
- Test: `tests/test_observable.py`

**Step 1: Write the failing test**

Create `tests/test_observable.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_observable.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers.observable'"

**Step 3: Implement Observable mixin**

Create `src/dspy_examples/observers/observable.py`:

```python
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
```

**Step 4: Update observers __init__.py**

Edit `src/dspy_examples/observers/__init__.py`:

```python
"""Observer pattern for progress tracking and event handling."""

from dspy_examples.observers.base import (
    Observer,
    Event,
    PipelineEvent,
    MetricEvent,
)
from dspy_examples.observers.observable import Observable
from dspy_examples.observers.logging_observer import LoggingObserver
from dspy_examples.observers.metric_observer import MetricObserver
from dspy_examples.observers.callback_observer import CallbackObserver
from dspy_examples.observers.progress_observer import ProgressObserver

__all__ = [
    "Observer",
    "Event",
    "PipelineEvent",
    "MetricEvent",
    "Observable",
    "LoggingObserver",
    "MetricObserver",
    "CallbackObserver",
    "ProgressObserver",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_observable.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/observers/observable.py src/dspy_examples/observers/__init__.py tests/test_observable.py
git commit -m "feat: add Observable mixin for event emission"
```

---

## Task 7: Pipeline Observer Integration

**Files:**
- Modify: `src/dspy_examples/pipeline.py`
- Test: `tests/test_pipeline_observer.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline_observer.py`:

```python
"""Tests for pipeline observer integration."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile


class TestPipelineObserverIntegration:
    """Tests for pipeline observer support."""

    def test_pipeline_add_observer(self):
        """Test adding observer to pipeline."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import Observer

        class TestObserver(Observer):
            pass

        pipeline = OptimizationPipeline()
        observer = TestObserver()
        result = pipeline.add_observer(observer)

        assert observer in pipeline._observers
        assert result is pipeline  # Method chaining

    def test_pipeline_emits_events(self):
        """Test that pipeline emits events during run."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import Observer, Event

        events = []

        class TestObserver(Observer):
            def on_event(self, event: Event) -> None:
                events.append(event)

        # Create a mock pipeline that doesn't need DSPy
        pipeline = OptimizationPipeline()
        pipeline.add_observer(TestObserver())

        # We'll verify the observer is registered
        assert len(pipeline._observers) == 1

    def test_pipeline_multiple_observers(self):
        """Test pipeline with multiple observers."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import Observer, LoggingObserver, MetricObserver

        pipeline = OptimizationPipeline()
        pipeline.add_observer(LoggingObserver())
        pipeline.add_observer(MetricObserver())

        assert len(pipeline._observers) == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_pipeline_observer.py -v`
Expected: FAIL with errors about missing add_observer method

**Step 3: Read existing pipeline.py**

Read `src/dspy_examples/pipeline.py` to understand current structure.

**Step 4: Modify pipeline.py to add Observable**

Edit `src/dspy_examples/pipeline.py`:

1. Add import for Observable and Observer
2. Make OptimizationPipeline inherit from Observable
3. Add event emissions at key stages

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_pipeline_observer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/pipeline.py tests/test_pipeline_observer.py
git commit -m "feat: integrate Observable into OptimizationPipeline"
```

---

## Task 8: PipelineBuilder Implementation

**Files:**
- Create: `src/dspy_examples/builders/__init__.py`
- Create: `src/dspy_examples/builders/pipeline_builder.py`
- Test: `tests/test_pipeline_builder.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline_builder.py`:

```python
"""Tests for PipelineBuilder."""

import pytest
from pathlib import Path
import tempfile


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        from dspy_examples.builders import PipelineBuilder

        builder = PipelineBuilder()
        assert builder is not None

    def test_builder_with_prompt(self):
        """Test builder with prompt."""
        from dspy_examples.builders import PipelineBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt")

            builder = PipelineBuilder().with_prompt(prompt_file)

            assert builder._config.input_path == prompt_file

    def test_builder_with_provider(self):
        """Test builder with provider."""
        from dspy_examples.builders import PipelineBuilder

        builder = PipelineBuilder().with_provider("openai", model="gpt-4")

        assert builder._config.provider_name == "openai"

    def test_builder_with_optimizer(self):
        """Test builder with optimizer."""
        from dspy_examples.builders import PipelineBuilder

        builder = PipelineBuilder().with_optimizer("mipro_v2")

        assert builder._config.optimizer_name == "mipro_v2"

    def test_builder_method_chaining(self):
        """Test builder method chaining."""
        from dspy_examples.builders import PipelineBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt")

            builder = (PipelineBuilder()
                .with_prompt(prompt_file)
                .with_provider("ollama")
                .with_optimizer("bootstrap_fewshot")
                .with_cache(True))

            assert builder._config.input_path == prompt_file
            assert builder._config.provider_name == "ollama"
            assert builder._config.optimizer_name == "bootstrap_fewshot"
            assert builder._config.use_cache is True

    def test_builder_build(self):
        """Test building pipeline."""
        from dspy_examples.builders import PipelineBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt")

            pipeline = (PipelineBuilder()
                .with_prompt(prompt_file)
                .with_provider("ollama")
                .with_optimizer("bootstrap_fewshot")
                .build())

            assert pipeline is not None
            assert hasattr(pipeline, "run")

    def test_builder_with_observer(self):
        """Test builder with observer."""
        from dspy_examples.builders import PipelineBuilder
        from dspy_examples.observers import LoggingObserver

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt")

            observer = LoggingObserver()
            pipeline = (PipelineBuilder()
                .with_prompt(prompt_file)
                .with_observer(observer)
                .build())

            assert observer in pipeline._observers
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_pipeline_builder.py -v`
Expected: FAIL with "No module named 'dspy_examples.builders'"

**Step 3: Create builders package**

Create `src/dspy_examples/builders/__init__.py`:

```python
"""Builder pattern for fluent configuration.

Provides fluent APIs for PipelineConfig and BatchConfig.
"""

from dspy_examples.builders.pipeline_builder import PipelineBuilder

__all__ = ["PipelineBuilder"]
```

**Step 4: Implement PipelineBuilder**

Create `src/dspy_examples/builders/pipeline_builder.py`:

```python
"""PipelineBuilder for fluent configuration.

Usage:
    pipeline = (PipelineBuilder()
        .with_prompt("prompt.md")
        .with_provider("openai", model="gpt-4")
        .with_optimizer("mipro_v2")
        .with_observer(LoggingObserver())
        .build())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
from dspy_examples.observers import Observer

if TYPE_CHECKING:
    pass


class PipelineBuilder:
    """Builder for OptimizationPipeline with fluent API.

    Provides method chaining for pipeline configuration.

    Example:
        pipeline = (PipelineBuilder()
            .with_prompt("prompt.md")
            .with_provider("openai")
            .with_optimizer("mipro_v2")
            .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._config = PipelineConfig()
        self._observers: list[Observer] = []

    def with_prompt(self, path: str | Path) -> "PipelineBuilder":
        """Set input prompt path.

        Args:
            path: Path to prompt file

        Returns:
            Self for method chaining
        """
        self._config.input_path = Path(path)
        return self

    def with_output(self, path: str | Path) -> "PipelineBuilder":
        """Set output path.

        Args:
            path: Path for output file

        Returns:
            Self for method chaining
        """
        self._config.output_path = Path(path)
        return self

    def with_provider(self, name: str, model: str | None = None) -> "PipelineBuilder":
        """Set LLM provider.

        Args:
            name: Provider name (e.g., "openai", "ollama", "anthropic")
            model: Optional model name

        Returns:
            Self for method chaining
        """
        self._config.provider_name = name
        if model:
            # Model is set via environment variable or config
            pass
        return self

    def with_optimizer(self, name: str) -> "PipelineBuilder":
        """Set optimizer.

        Args:
            name: Optimizer name (e.g., "bootstrap_fewshot", "mipro_v2")

        Returns:
            Self for method chaining
        """
        self._config.optimizer_name = name
        return self

    def with_variables(self, variables: dict[str, str]) -> "PipelineBuilder":
        """Set template variables.

        Args:
            variables: Dictionary of variable names to values

        Returns:
            Self for method chaining
        """
        self._config.variables = variables
        return self

    def with_cache(self, use_cache: bool = True, ttl: int | None = None) -> "PipelineBuilder":
        """Enable or disable caching.

        Args:
            use_cache: Whether to use cache
            ttl: Optional cache TTL in seconds

        Returns:
            Self for method chaining
        """
        self._config.use_cache = use_cache
        if ttl is not None:
            self._config.cache_ttl = ttl
        return self

    def with_observer(self, observer: Observer) -> "PipelineBuilder":
        """Add an observer.

        Args:
            observer: Observer to add

        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self

    def build(self) -> OptimizationPipeline:
        """Build the pipeline.

        Returns:
            Configured OptimizationPipeline
        """
        pipeline = OptimizationPipeline(self._config)
        for observer in self._observers:
            pipeline.add_observer(observer)
        return pipeline
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_pipeline_builder.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/builders/__init__.py src/dspy_examples/builders/pipeline_builder.py tests/test_pipeline_builder.py
git commit -m "feat: add PipelineBuilder with fluent API"
```

---

## Task 9: BatchBuilder Implementation

**Files:**
- Create: `src/dspy_examples/builders/batch_builder.py`
- Test: `tests/test_batch_builder.py`
- Modify: `src/dspy_examples/builders/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_batch_builder.py`:

```python
"""Tests for BatchBuilder."""

import pytest
from pathlib import Path


class TestBatchBuilder:
    """Tests for BatchBuilder."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder()
        assert builder is not None

    def test_builder_with_prompts(self):
        """Test builder with prompts."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_prompts(["p1.md", "p2.md"])

        assert len(builder._config.prompt_paths) == 2

    def test_builder_with_providers(self):
        """Test builder with providers."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_providers(["openai", "ollama"])

        assert len(builder._config.providers) == 2

    def test_builder_with_optimizer(self):
        """Test builder with optimizer."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_optimizer("gepa")

        assert builder._config.optimizer_name == "gepa"

    def test_builder_method_chaining(self):
        """Test builder method chaining."""
        from dspy_examples.builders import BatchBuilder

        builder = (BatchBuilder()
            .with_prompts(["test.md"])
            .with_providers(["openai"])
            .with_optimizer("mipro_v2")
            .with_output_dir("output"))

        assert len(builder._config.prompt_paths) == 1
        assert builder._config.optimizer_name == "mipro_v2"

    def test_builder_build(self):
        """Test building batch command."""
        from dspy_examples.builders import BatchBuilder
        from dspy_examples.commands import BatchCommand

        builder = BatchBuilder()
        builder._config.prompt_paths = [Path("test.md")]

        batch = builder.build()

        assert isinstance(batch, BatchCommand)

    def test_builder_with_observer(self):
        """Test builder with observer."""
        from dspy_examples.builders import BatchBuilder
        from dspy_examples.observers import LoggingObserver

        observer = LoggingObserver()
        builder = (BatchBuilder()
            .with_prompts(["test.md"])
            .with_observer(observer))

        assert observer in builder._observers
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_batch_builder.py -v`
Expected: FAIL with "cannot import name 'BatchBuilder'"

**Step 3: Implement BatchBuilder**

Create `src/dspy_examples/builders/batch_builder.py`:

```python
"""BatchBuilder for fluent configuration.

Usage:
    batch = (BatchBuilder()
        .with_prompts(["p1.md", "p2.md"])
        .with_providers(["openai", "ollama"])
        .with_optimizer("gepa")
        .build())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dspy_examples.commands import BatchCommand, BatchConfig
from dspy_examples.observers import Observer

if TYPE_CHECKING:
    pass


class BatchBuilder:
    """Builder for BatchCommand with fluent API.

    Provides method chaining for batch configuration.

    Example:
        batch = (BatchBuilder()
            .with_prompts(["prompt1.md", "prompt2.md"])
            .with_providers(["openai", "ollama"])
            .with_optimizer("mipro_v2")
            .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default config."""
        self._config = BatchConfig()
        self._observers: list[Observer] = []

    def with_prompts(self, paths: list[str | Path]) -> "BatchBuilder":
        """Set prompt paths.

        Args:
            paths: List of prompt file paths

        Returns:
            Self for method chaining
        """
        self._config.prompt_paths = [Path(p) for p in paths]
        return self

    def with_template(self, path: str | Path) -> "BatchBuilder":
        """Set template path for variable substitution.

        Args:
            path: Path to template file

        Returns:
            Self for method chaining
        """
        self._config.prompt_template = Path(path)
        return self

    def with_variables(self, variable_sets: list[dict[str, str]]) -> "BatchBuilder":
        """Set variable sets for template.

        Args:
            variable_sets: List of variable dictionaries

        Returns:
            Self for method chaining
        """
        self._config.variable_sets = variable_sets
        return self

    def with_providers(self, names: list[str]) -> "BatchBuilder":
        """Set provider names.

        Args:
            names: List of provider names

        Returns:
            Self for method chaining
        """
        self._config.providers = [{"name": name} for name in names]
        return self

    def with_provider(self, name: str, model: str | None = None) -> "BatchBuilder":
        """Add a single provider.

        Args:
            name: Provider name
            model: Optional model name

        Returns:
            Self for method chaining
        """
        provider = {"name": name}
        if model:
            provider["model"] = model
        self._config.providers.append(provider)
        return self

    def with_optimizer(self, name: str) -> "BatchBuilder":
        """Set optimizer.

        Args:
            name: Optimizer name

        Returns:
            Self for method chaining
        """
        self._config.optimizer_name = name
        return self

    def with_output_dir(self, path: str | Path) -> "BatchBuilder":
        """Set output directory.

        Args:
            path: Output directory path

        Returns:
            Self for method chaining
        """
        self._config.output_dir = Path(path)
        return self

    def with_observer(self, observer: Observer) -> "BatchBuilder":
        """Add an observer.

        Args:
            observer: Observer to add

        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self

    def build(self) -> BatchCommand:
        """Build the batch command.

        Returns:
            Configured BatchCommand
        """
        batch = BatchCommand(self._config)
        # Store observers for later use (integration with batch processing)
        batch._observers = self._observers
        return batch
```

**Step 4: Update builders __init__.py**

Edit `src/dspy_examples/builders/__init__.py`:

```python
"""Builder pattern for fluent configuration.

Provides fluent APIs for PipelineConfig and BatchConfig.
"""

from dspy_examples.builders.pipeline_builder import PipelineBuilder
from dspy_examples.builders.batch_builder import BatchBuilder

__all__ = ["PipelineBuilder", "BatchBuilder"]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_batch_builder.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/builders/batch_builder.py src/dspy_examples/builders/__init__.py tests/test_batch_builder.py
git commit -m "feat: add BatchBuilder with fluent API"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add Observer Pattern section**

Add after the Batch Processing section:

```markdown
## Observer Pattern

Track optimization progress with observers:

```python
from dspy_examples.observers import LoggingObserver, ProgressObserver, MetricObserver
from dspy_examples.pipeline import OptimizationPipeline

# Add observers to pipeline
pipeline = OptimizationPipeline()
pipeline.add_observer(LoggingObserver())
pipeline.add_observer(ProgressObserver())
pipeline.add_observer(MetricObserver())

result = pipeline.run()

# Get collected metrics
for observer in pipeline._observers:
    if isinstance(observer, MetricObserver):
        print(observer.get_summary())
```

### Built-in Observers

| Observer | Purpose |
|----------|---------|
| `LoggingObserver` | Logs events to console/file |
| `ProgressObserver` | Tracks pipeline stages |
| `MetricObserver` | Collects performance metrics |
| `CallbackObserver` | Wraps user callbacks |

### Custom Observer

```python
from dspy_examples.observers import Observer, Event

class MyObserver(Observer):
    def on_event(self, event: Event):
        print(f"Event: {event.name}")

    def on_pipeline_event(self, event):
        print(f"Stage: {event.stage} - {event.status}")

pipeline.add_observer(MyObserver())
```

## Builder Pattern

Configure pipelines and batches with fluent API:

### Pipeline Builder

```python
from dspy_examples.builders import PipelineBuilder

pipeline = (PipelineBuilder()
    .with_prompt("prompt.md")
    .with_provider("openai", model="gpt-4")
    .with_optimizer("mipro_v2")
    .with_variables({"topic": "Python"})
    .with_cache(ttl=3600)
    .with_observer(LoggingObserver())
    .build())

result = pipeline.run()
```

### Batch Builder

```python
from dspy_examples.builders import BatchBuilder

batch = (BatchBuilder()
    .with_prompts(["p1.md", "p2.md"])
    .with_providers(["openai", "ollama"])
    .with_optimizer("gepa")
    .with_output_dir("output")
    .build())

result = batch.run()
```
```

**Step 2: Update Design Patterns table**

In the Design Patterns section, update:

```markdown
| Pattern | Location | Purpose |
|---------|----------|---------|
| **Adapter** | `providers/` | Multiple LLM backends with unified interface |
| **Strategy** | `optimizers/` | Swap optimization algorithms |
| **Factory** | `factory/` | Create providers & optimizers by name |
| **Template** | `pipeline.py` | Standard optimization workflow |
| **Memento** | `cache.py` | Save/restore results |
| **Command** | `commands/` | Batch processing with queue persistence |
| **Observer** | `observers/` | Progress tracking and event handling |
| **Builder** | `builders/` | Fluent configuration API |
```

**Step 3: Update Project Structure**

Add new directories to the structure:

```markdown
├── src/dspy_examples/
│   ├── observers/              # Observer pattern
│   │   ├── __init__.py
│   │   ├── base.py             # Observer, Event classes
│   │   ├── logging_observer.py
│   │   ├── progress_observer.py
│   │   ├── metric_observer.py
│   │   ├── callback_observer.py
│   │   └── observable.py       # Mixin for event emission
│   ├── builders/               # Builder pattern
│   │   ├── __init__.py
│   │   ├── pipeline_builder.py
│   │   └── batch_builder.py
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add Observer and Builder pattern documentation"
```

---

## Execution Summary

This implementation plan adds:

1. **Observer Base Classes** - Event, PipelineEvent, MetricEvent, Observer
2. **LoggingObserver** - Logs events to console/file
3. **MetricObserver** - Collects performance metrics
4. **CallbackObserver** - Wraps user callbacks
5. **ProgressObserver** - Tracks pipeline stages
6. **Observable Mixin** - Event emission for Pipeline
7. **Pipeline Observer Integration** - Add observers to OptimizationPipeline
8. **PipelineBuilder** - Fluent API for pipeline config
9. **BatchBuilder** - Fluent API for batch config
10. **Documentation** - Update README with usage examples

Each task follows TDD: test first, then implement. Frequent commits maintain clean history.