"""Tests for command base classes."""

import pytest
from datetime import datetime
from pathlib import Path


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_command_result_creation(self):
        """Test creating a successful result."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_001",
            status="success",
            output_path=Path("output/prompt1.md"),
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            model_name="llama3",
            execution_time=42.5,
        )

        assert result.command_id == "cmd_001"
        assert result.status == "success"
        assert result.output_path == Path("output/prompt1.md")
        assert result.optimizer_name == "bootstrap_fewshot"
        assert result.provider_name == "ollama"
        assert result.model_name == "llama3"
        assert result.execution_time == 42.5
        assert result.error_message is None
        assert result.metadata == {}

    def test_command_result_failed(self):
        """Test creating a failed result."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_002",
            status="failed",
            output_path=None,
            optimizer_name="mipro_v2",
            provider_name="openai",
            model_name="gpt-4",
            execution_time=5.2,
            error_message="Connection timeout",
        )

        assert result.status == "failed"
        assert result.error_message == "Connection timeout"
        assert result.output_path is None

    def test_command_result_with_metadata(self):
        """Test result with additional metadata."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_003",
            status="success",
            output_path=Path("output/prompt2.md"),
            optimizer_name="gepa",
            provider_name="anthropic",
            model_name="claude-3",
            execution_time=30.0,
            metadata={"tokens_used": 1500, "attempts": 2},
        )

        assert result.metadata["tokens_used"] == 1500
        assert result.metadata["attempts"] == 2


class TestCommand:
    """Tests for Command abstract base class."""

    def test_command_abstract_methods(self):
        """Test that Command requires exec implementation."""
        from dspy_examples.commands.base import Command

        with pytest.raises(TypeError):
            Command()  # Cannot instantiate abstract class

    def test_command_implementation(self):
        """Test implementing a concrete command."""
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, command_id: str):
                self._id = command_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=Path("test.md"),
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=1.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        cmd = TestCommand("test_001")
        result = cmd.execute()

        assert result.command_id == "test_001"
        assert result.status == "success"

    def test_command_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from dspy_examples.commands.base import Command, CommandResult

        class SerializeTestCommand(Command):
            def __init__(self, cmd_id: str, value: str):
                self._id = cmd_id
                self._value = value

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
                return {"id": self._id, "value": self._value}

            @classmethod
            def from_dict(cls, data: dict) -> "SerializeTestCommand":
                return cls(data["id"], data["value"])

        cmd1 = SerializeTestCommand("id_001", "hello")
        data = cmd1.to_dict()
        cmd2 = SerializeTestCommand.from_dict(data)

        assert cmd2.command_id == "id_001"
        assert cmd2._value == "hello"