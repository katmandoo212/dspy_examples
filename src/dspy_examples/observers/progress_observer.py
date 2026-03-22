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

    def on_event(self, event: Event) -> None:
        """Handle any event.

        Args:
            event: The event that was emitted
        """
        pass

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Track pipeline stage progress."""
        # Track stages based on status
        if event.status == "started":
            # A stage started - just update current stage
            self.current_stage = event.stage
        elif event.status == "completed":
            # A stage completed - increment counter
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