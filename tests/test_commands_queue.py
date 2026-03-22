"""Tests for SQLite-backed command queue."""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_db_path():
    """Create a temporary database path that properly cleans up on Windows."""
    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "queue.db"
    yield db_path
    # Cleanup - try multiple times on Windows
    import time
    for _ in range(3):
        try:
            if db_path.exists():
                os.unlink(db_path)
            os.rmdir(tmpdir)
            break
        except PermissionError:
            time.sleep(0.1)


class TestCommandQueue:
    """Tests for SQLite command queue."""

    def test_queue_creation(self, temp_db_path):
        """Test creating a new queue database."""
        from dspy_examples.commands.queue import CommandQueue

        queue = CommandQueue(temp_db_path)

        assert temp_db_path.exists()
        assert queue.is_empty()

    def test_add_command(self, temp_db_path):
        """Test adding a command to the queue."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class SimpleCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "simple"}

            @classmethod
            def from_dict(cls, data: dict) -> "SimpleCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        cmd = SimpleCommand("cmd_001")
        queue.add(cmd)

        assert queue.size() == 1
        assert not queue.is_empty()

    def test_get_pending(self, temp_db_path):
        """Test retrieving pending commands."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        queue.add(TestCommand("cmd_001"))
        queue.add(TestCommand("cmd_002"))

        pending = queue.get_pending()
        assert len(pending) == 2
        assert pending[0][0] == "cmd_001"  # Returns tuple (id, data)
        assert pending[1][0] == "cmd_002"

    def test_mark_completed(self, temp_db_path):
        """Test marking a command as completed."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        queue.add(TestCommand("cmd_001"))
        queue.add(TestCommand("cmd_002"))

        queue.mark_completed("cmd_001")

        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0][0] == "cmd_002"

    def test_mark_failed(self, temp_db_path):
        """Test marking a command as failed."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        queue.add(TestCommand("cmd_001"))
        queue.mark_failed("cmd_001", "Connection error")

        pending = queue.get_pending()
        assert len(pending) == 0

        failed = queue.get_failed()
        assert len(failed) == 1
        assert failed[0]["id"] == "cmd_001"
        assert failed[0]["error"] == "Connection error"

    def test_get_completed(self, temp_db_path):
        """Test retrieving completed commands."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        queue.add(TestCommand("cmd_001"))
        queue.add(TestCommand("cmd_002"))
        queue.mark_completed("cmd_001")

        completed = queue.get_completed()
        assert len(completed) == 1
        assert completed[0]["id"] == "cmd_001"

    def test_clear_queue(self, temp_db_path):
        """Test clearing all commands from queue."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        queue = CommandQueue(temp_db_path)

        queue.add(TestCommand("cmd_001"))
        queue.add(TestCommand("cmd_002"))

        queue.clear()

        assert queue.is_empty()
        assert queue.size() == 0