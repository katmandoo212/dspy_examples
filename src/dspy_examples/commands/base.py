"""Base classes for command pattern.

Command pattern encapsulates optimization requests as objects,
enabling queue-based execution, persistence, and batch processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class CommandResult:
    """Result of executing a command.

    Attributes:
        command_id: Unique identifier for the command
        status: Execution status (success, failed, skipped)
        output_path: Path to output file (None if failed)
        optimizer_name: Name of optimizer used
        provider_name: Name of LLM provider
        model_name: Name of model used
        execution_time: Time in seconds for execution
        error_message: Error message if failed (None if success)
        metadata: Additional metadata about execution
    """

    command_id: str
    status: Literal["success", "failed", "skipped"]
    output_path: Path | None
    optimizer_name: str
    provider_name: str
    model_name: str
    execution_time: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Command(ABC):
    """Abstract base class for commands.

    Commands encapsulate optimization requests that can be:
    - Queued for execution
    - Persisted to SQLite for recovery
    - Executed by workers
    """

    @property
    @abstractmethod
    def command_id(self) -> str:
        """Unique identifier for this command."""
        pass

    @abstractmethod
    def execute(self) -> CommandResult:
        """Execute this command and return the result."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize command to dictionary for persistence."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        """Deserialize command from dictionary."""
        pass
