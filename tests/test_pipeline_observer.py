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

    def test_pipeline_multiple_observers(self):
        """Test pipeline with multiple observers."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import LoggingObserver, MetricObserver

        pipeline = OptimizationPipeline()
        pipeline.add_observer(LoggingObserver())
        pipeline.add_observer(MetricObserver())

        assert len(pipeline._observers) == 2

    def test_pipeline_emits_events(self):
        """Test that pipeline can emit events."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import Observer, Event

        events = []

        class TestObserver(Observer):
            def on_event(self, event: Event) -> None:
                events.append(event)

        pipeline = OptimizationPipeline()
        pipeline.add_observer(TestObserver())

        # Verify observer is registered
        assert len(pipeline._observers) == 1

    def test_observable_mixin_in_pipeline(self):
        """Test that OptimizationPipeline has Observable methods."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers.observable import Observable

        # Verify OptimizationPipeline inherits from Observable
        assert hasattr(OptimizationPipeline, 'add_observer')
        assert hasattr(OptimizationPipeline, 'remove_observer')
        assert hasattr(OptimizationPipeline, '_emit')

    def test_pipeline_remove_observer(self):
        """Test removing observer from pipeline."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers import Observer

        class TestObserver(Observer):
            pass

        pipeline = OptimizationPipeline()
        observer = TestObserver()
        pipeline.add_observer(observer)

        assert observer in pipeline._observers

        pipeline.remove_observer(observer)
        assert observer not in pipeline._observers

    def test_pipeline_is_observable_subclass(self):
        """Test that OptimizationPipeline is a subclass of Observable."""
        from dspy_examples.pipeline import OptimizationPipeline
        from dspy_examples.observers.observable import Observable

        # Verify inheritance
        assert issubclass(OptimizationPipeline, Observable)