# Observer and Builder Patterns - Design Document

**Date**: 2026-03-22
**Status**: Approved for Implementation

## Summary

Add Observer Pattern for progress tracking and Builder Pattern for cleaner configuration APIs. The Observer Pattern provides extensible event system with full observability by default. The Builder Pattern enables fluent API with method chaining for PipelineConfig and BatchConfig.

## Observer Pattern

### Purpose

Track optimization progress with:
- Pipeline lifecycle events (started, step_started, step_completed, completed, failed)
- Performance metrics (tokens_used, duration, attempts, cache_hits)
- Extensible system for custom events and user data

### Core Architecture

**Event Types:**
- `PipelineEvent` - lifecycle events with stage, status, duration
- `MetricEvent` - performance data with value, unit, aggregation

**Observer Interface:**
```python
class Observer(ABC):
    def on_event(self, event: Event) -> None:
        """Called when any event is emitted."""
        pass

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Called for pipeline lifecycle events."""
        pass

    def on_metric_event(self, event: MetricEvent) -> None:
        """Called for metric events."""
        pass
```

**Built-in Observers:**
- `LoggingObserver` - logs events to console/file
- `ProgressObserver` - shows progress bar (tqdm integration)
- `MetricObserver` - collects metrics for analysis
- `CallbackObserver` - wraps user-provided callback functions

### Event Data Structure

**Base Event:**
```python
@dataclass
class Event:
    name: str                    # Event name (e.g., "optimization_started")
    timestamp: datetime           # When event occurred
    source: str                   # Source component (e.g., "pipeline", "optimizer")
    data: dict[str, Any]          # Event-specific data
    metadata: dict[str, Any] = field(default_factory=dict)  # User-extensible data
```

**Pipeline Events:**
```python
@dataclass
class PipelineEvent(Event):
    stage: str              # "load_prompt", "configure", "optimize", "save"
    status: str             # "started", "completed", "failed"
    duration_ms: int | None = None
    error: str | None = None
```

**Metric Events:**
```python
@dataclass
class MetricEvent(Event):
    value: float            # The metric value
    unit: str               # "tokens", "seconds", "attempts"
    aggregation: str = "sum"  # How to aggregate: "sum", "max", "last"
```

### Integration Points

- `Pipeline.run()` emits events at each stage
- `Optimizer.optimize()` emits step events
- `BatchCommand.run()` emits batch progress events

### Usage Example

```python
from dspy_examples.observers import LoggingObserver, ProgressObserver, MetricObserver

pipeline = OptimizationPipeline()
pipeline.add_observer(LoggingObserver())
pipeline.add_observer(ProgressObserver())

# Custom observer
class MyObserver(Observer):
    def on_metric_event(self, event):
        if event.name == "tokens_used":
            print(f"Tokens: {event.value}")

pipeline.add_observer(MyObserver())

result = pipeline.run()
```

## Builder Pattern

### Purpose

Provide cleaner API for:
- Building complex PipelineConfig with fluent interface
- Building BatchConfig with method chaining
- Composing multiple optimizers/providers easily

### PipelineBuilder

```python
from dspy_examples.builders import PipelineBuilder

# Current verbose approach
config = PipelineConfig(
    provider_name="openai",
    optimizer_name="mipro_v2",
    input_path=Path("prompt.md"),
    output_path=Path("optimized.md"),
    variables={"topic": "Python"},
    use_cache=True,
    cache_ttl=3600,
)
pipeline = OptimizationPipeline(config)

# New fluent API
pipeline = (PipelineBuilder()
    .with_prompt("prompt.md")
    .with_provider("openai", model="gpt-4")
    .with_optimizer("mipro_v2")
    .with_variables({"topic": "Python"})
    .with_cache(ttl=3600)
    .with_output("optimized.md")
    .build())
```

### BatchBuilder

```python
from dspy_examples.builders import BatchBuilder

# Current verbose approach
config = BatchConfig(
    prompt_paths=[Path("p1.md"), Path("p2.md")],
    providers=[{"name": "openai"}, {"name": "ollama"}],
    optimizer_name="bootstrap_fewshot",
    output_dir=Path("output"),
)

# New fluent API
batch = (BatchBuilder()
    .with_prompts(["p1.md", "p2.md"])
    .with_providers(["openai", "ollama"])
    .with_optimizer("bootstrap_fewshot")
    .with_output_dir("output")
    .with_observer(LoggingObserver())
    .build())
```

### Composability

**Multiple Providers:**
```python
pipeline = (PipelineBuilder()
    .with_prompt("prompt.md")
    .with_providers(["openai", "anthropic"])  # Try multiple providers
    .with_optimizer("gepa")
    .build())
```

**Chained Optimizers:**
```python
pipeline = (PipelineBuilder()
    .with_prompt("prompt.md")
    .with_optimizer("bootstrap_fewshot")
    .then("mipro_v2")  # Chain optimizers
    .build())
```

**With Observers:**
```python
pipeline = (PipelineBuilder()
    .with_prompt("prompt.md")
    .with_provider("ollama")
    .with_observer(LoggingObserver())
    .with_observer(ProgressObserver())
    .build())
```

## File Structure

**New Files:**
```
src/dspy_examples/
├── observers/
│   ├── __init__.py
│   ├── base.py              # Observer, Event, PipelineEvent, MetricEvent
│   ├── logging_observer.py  # Logs events to console/file
│   ├── progress_observer.py # Progress bar with tqdm
│   ├── metric_observer.py   # Collects metrics for analysis
│   └── callback_observer.py # Wraps user callbacks
├── builders/
│   ├── __init__.py
│   ├── pipeline_builder.py  # Fluent API for PipelineConfig
│   └── batch_builder.py     # Fluent API for BatchConfig
```

**Modified Files:**
```
src/dspy_examples/
├── pipeline.py              # Add observer management
├── commands/
│   └── batch.py             # Add observer support to BatchCommand
```

## Implementation Tasks

### Task 1: Observer Base Classes

**Files:**
- Create: `src/dspy_examples/observers/__init__.py`
- Create: `src/dspy_examples/observers/base.py`
- Test: `tests/test_observers_base.py`

**Step 1: Write failing tests for Event classes**

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
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_base.py -v`
Expected: FAIL with "No module named 'dspy_examples.observers'"

**Step 3: Implement observer base classes**

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

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_observers_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dspy_examples/observers/__init__.py src/dspy_examples/observers/base.py tests/test_observers_base.py
git commit -m "feat: add observer base classes with Event, PipelineEvent, MetricEvent"
```

---

### Task 2: Built-in Observers

**Files:**
- Create: `src/dspy_examples/observers/logging_observer.py`
- Create: `src/dspy_examples/observers/progress_observer.py`
- Create: `src/dspy_examples/observers/metric_observer.py`
- Create: `src/dspy_examples/observers/callback_observer.py`
- Test: `tests/test_observers_builtin.py`

Implement LoggingObserver (console/file logging), ProgressObserver (tqdm), MetricObserver (metric collection), and CallbackObserver (user callbacks).

---

### Task 3: Pipeline Observer Integration

**Files:**
- Modify: `src/dspy_examples/pipeline.py`
- Test: `tests/test_pipeline_observer.py`

Add observer management to OptimizationPipeline:
- `add_observer()` method
- `_emit()` helper
- Event emissions at each stage

---

### Task 4: Builder Pattern for Pipeline

**Files:**
- Create: `src/dspy_examples/builders/__init__.py`
- Create: `src/dspy_examples/builders/pipeline_builder.py`
- Test: `tests/test_pipeline_builder.py`

Implement PipelineBuilder with fluent API.

---

### Task 5: Builder Pattern for Batch

**Files:**
- Create: `src/dspy_examples/builders/batch_builder.py`
- Test: `tests/test_batch_builder.py`

Implement BatchBuilder with fluent API.

---

### Task 6: Update Documentation

**Files:**
- Modify: `README.md`
- Update design patterns table
- Add Observer and Builder usage examples

---

## Design Patterns Update

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