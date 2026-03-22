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
                name="stage_completed",
                timestamp=datetime.now(),
                source="pipeline",
                data={},
                stage=stage,
                status="completed",
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