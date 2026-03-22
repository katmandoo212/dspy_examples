"""Tests for BatchBuilder."""

import pytest
from pathlib import Path


class TestBatchBuilder:
    """Tests for BatchBuilder."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder()
        assert builder is not None

    def test_builder_with_prompts(self):
        """Test builder with prompts."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_prompts(["p1.md", "p2.md"])

        assert len(builder._config.prompt_paths) == 2

    def test_builder_with_providers(self):
        """Test builder with providers."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_providers(["openai", "ollama"])

        assert len(builder._config.providers) == 2

    def test_builder_with_optimizer(self):
        """Test builder with optimizer."""
        from dspy_examples.builders import BatchBuilder

        builder = BatchBuilder().with_optimizer("gepa")

        assert builder._config.optimizer_name == "gepa"

    def test_builder_method_chaining(self):
        """Test builder method chaining."""
        from dspy_examples.builders import BatchBuilder

        builder = (BatchBuilder()
            .with_prompts(["test.md"])
            .with_providers(["openai"])
            .with_optimizer("mipro_v2")
            .with_output_dir("output"))

        assert len(builder._config.prompt_paths) == 1
        assert builder._config.optimizer_name == "mipro_v2"

    def test_builder_build(self):
        """Test building batch command."""
        from dspy_examples.builders import BatchBuilder
        from dspy_examples.commands import BatchCommand

        builder = BatchBuilder()
        builder._config.prompt_paths = [Path("test.md")]

        batch = builder.build()

        assert isinstance(batch, BatchCommand)

    def test_builder_with_observer(self):
        """Test builder with observer."""
        from dspy_examples.builders import BatchBuilder
        from dspy_examples.observers import LoggingObserver

        observer = LoggingObserver()
        builder = (BatchBuilder()
            .with_prompts(["test.md"])
            .with_observer(observer))

        assert observer in builder._observers