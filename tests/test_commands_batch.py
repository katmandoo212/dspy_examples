"""Tests for high-level BatchCommand API."""

import pytest
from pathlib import Path
import tempfile
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory that properly cleans up on Windows."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Cleanup
    import time
    for _ in range(3):
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            break
        except PermissionError:
            time.sleep(0.1)


class TestBatchCommand:
    """Tests for BatchCommand."""

    def test_batch_command_creation(self):
        """Test creating a BatchCommand with config."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/test.md")],
            providers=[{"name": "ollama"}],
        )

        batch = BatchCommand(config)

        assert batch.config == config
        assert batch.batch_id is not None

    def test_batch_command_generate_configs(self):
        """Test config generation for multi-prompt batch."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[
                Path("prompts/q1.md"),
                Path("prompts/q2.md"),
            ],
            providers=[{"name": "ollama"}],
            optimizer_name="bootstrap_fewshot",
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 2
        assert configs[0]["prompt_path"] == Path("prompts/q1.md")
        assert configs[1]["prompt_path"] == Path("prompts/q2.md")

    def test_batch_command_multi_provider(self):
        """Test config generation for multi-provider batch."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/test.md")],
            providers=[
                {"name": "openai", "model": "gpt-4"},
                {"name": "anthropic", "model": "claude-3"},
                {"name": "ollama", "model": "llama3"},
            ],
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 3
        provider_names = [c["provider_name"] for c in configs]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "ollama" in provider_names

    def test_batch_command_with_variables(self):
        """Test config generation with variable sets."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_template=Path("prompts/template.md"),
            variable_sets=[
                {"country": "France", "tone": "formal"},
                {"country": "Japan", "tone": "casual"},
            ],
            providers=[{"name": "ollama"}],
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 2
        assert configs[0]["variables"]["country"] == "France"
        assert configs[1]["variables"]["country"] == "Japan"

    def test_batch_command_get_output_paths(self):
        """Test output path generation."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/question.md")],
            providers=[
                {"name": "openai", "model": "gpt-4"},
                {"name": "ollama", "model": "llama3"},
            ],
            optimizer_name="mipro_v2",
            output_dir=Path("prompts/batch_output"),
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        # Check naming pattern
        assert "question_openai_gpt-4_mipro_v2.md" in str(configs[0]["output_path"])
        assert "question_ollama_llama3_mipro_v2.md" in str(configs[1]["output_path"])


class TestOptimizeCommand:
    """Tests for OptimizeCommand."""

    def test_optimize_command_creation(self):
        """Test creating an OptimizeCommand."""
        from dspy_examples.commands.batch import OptimizeCommand
        from dspy_examples.commands.base import CommandResult

        cmd = OptimizeCommand(
            command_id="test_001",
            prompt_path=Path("prompts/test.md"),
            output_path=Path("output/test.md"),
            provider_name="ollama",
            model_name="llama3",
            optimizer_name="bootstrap_fewshot",
            variables={},
        )

        assert cmd.command_id == "test_001"
        assert cmd.provider_name == "ollama"
        assert cmd.optimizer_name == "bootstrap_fewshot"

    def test_optimize_command_serialization(self):
        """Test OptimizeCommand serialization."""
        from dspy_examples.commands.batch import OptimizeCommand

        cmd = OptimizeCommand(
            command_id="test_001",
            prompt_path=Path("prompts/test.md"),
            output_path=Path("output/test.md"),
            provider_name="ollama",
            model_name="llama3",
            optimizer_name="bootstrap_fewshot",
            variables={"topic": "Python"},
        )

        data = cmd.to_dict()
        assert data["id"] == "test_001"
        assert data["provider_name"] == "ollama"
        assert data["variables"]["topic"] == "Python"

        # Round-trip
        cmd2 = OptimizeCommand.from_dict(data)
        assert cmd2.command_id == "test_001"
        assert cmd2.variables["topic"] == "Python"