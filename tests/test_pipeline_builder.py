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