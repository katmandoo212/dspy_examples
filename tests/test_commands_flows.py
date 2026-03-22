"""Tests for batch processing flows."""

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


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_batch_config_defaults(self):
        """Test default batch configuration."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig()

        assert config.providers == [{"name": "ollama"}]
        assert config.optimizer_name == "bootstrap_fewshot"
        assert config.output_dir == Path("prompts/batch_output")
        assert config.max_concurrent == 1
        assert config.retry_count == 2

    def test_batch_config_with_prompts(self):
        """Test batch config with prompt paths."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[
                Path("prompts/q1.md"),
                Path("prompts/q2.md"),
            ],
            providers=[{"name": "openai", "model": "gpt-4"}],
        )

        assert len(config.prompt_paths) == 2
        assert config.providers[0]["name"] == "openai"

    def test_batch_config_with_template(self):
        """Test batch config with template and variables."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_template=Path("prompts/template.md"),
            variable_sets=[
                {"country": "France", "tone": "formal"},
                {"country": "Japan", "tone": "casual"},
            ],
        )

        assert config.prompt_template == Path("prompts/template.md")
        assert len(config.variable_sets) == 2


class TestBatchResult:
    """Tests for BatchResult."""

    def test_batch_result_creation(self):
        """Test creating a batch result."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
            CommandResult(
                command_id="cmd_002",
                status="success",
                output_path=Path("output/p2.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="openai",
                model_name="gpt-4",
                execution_time=15.2,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=2,
            successful=2,
            failed=0,
            skipped=0,
            total_time=25.7,
            results=results,
        )

        assert batch_result.batch_id == "batch_001"
        assert batch_result.successful == 2
        assert batch_result.failed == 0

    def test_batch_result_to_markdown(self):
        """Test converting batch result to markdown."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
            by_provider={"ollama": {"count": 1, "avg_time": 10.5, "success_rate": 1.0, "model": "llama3"}},
            by_optimizer={"bootstrap_fewshot": {"count": 1, "avg_time": 10.5, "success_rate": 1.0}},
        )

        md = batch_result.to_markdown()

        assert "# Batch Report: batch_001" in md
        assert "Total: 1 commands" in md
        assert "Successful: 1" in md

    def test_batch_result_to_json(self):
        """Test converting batch result to JSON dict."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
        )

        json_dict = batch_result.to_json()

        assert json_dict["batch_id"] == "batch_001"
        assert json_dict["total_commands"] == 1
        assert json_dict["successful"] == 1

    def test_batch_result_save(self, temp_dir):
        """Test saving batch result to files."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
        )

        batch_result.save(temp_dir)

        # Check markdown file exists
        md_file = temp_dir / "batch_001_report.md"
        assert md_file.exists()

        # Check JSON file exists
        json_file = temp_dir / "batch_001_results.json"
        assert json_file.exists()


class TestResultsAggregator:
    """Tests for ResultsAggregator."""

    def test_aggregator_add_results(self):
        """Test adding results to aggregator."""
        from dspy_examples.commands.results import ResultsAggregator, CommandResult

        aggregator = ResultsAggregator()

        aggregator.add(CommandResult(
            command_id="cmd_001",
            status="success",
            output_path=Path("output/p1.md"),
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            model_name="llama3",
            execution_time=10.0,
        ))

        aggregator.add(CommandResult(
            command_id="cmd_002",
            status="failed",
            output_path=None,
            optimizer_name="mipro_v2",
            provider_name="openai",
            model_name="gpt-4",
            execution_time=5.0,
            error_message="Connection timeout",
        ))

        assert len(aggregator.results) == 2

    def test_aggregator_compute_statistics(self):
        """Test computing statistics from results."""
        from dspy_examples.commands.results import ResultsAggregator, CommandResult

        aggregator = ResultsAggregator()

        aggregator.add(CommandResult(
            command_id="cmd_001",
            status="success",
            output_path=Path("output/p1.md"),
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            model_name="llama3",
            execution_time=10.0,
        ))

        aggregator.add(CommandResult(
            command_id="cmd_002",
            status="success",
            output_path=Path("output/p2.md"),
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            model_name="llama3",
            execution_time=20.0,
        ))

        batch_result = aggregator.aggregate()

        assert batch_result.total_commands == 2
        assert batch_result.successful == 2
        assert batch_result.failed == 0
        assert batch_result.total_time == 30.0

        # Check by_provider statistics
        assert "ollama" in batch_result.by_provider
        assert batch_result.by_provider["ollama"]["count"] == 2
        assert batch_result.by_provider["ollama"]["avg_time"] == 15.0

        # Check by_optimizer statistics
        assert "bootstrap_fewshot" in batch_result.by_optimizer