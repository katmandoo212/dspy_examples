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

    def test_builder_with_provider_model(self):
        """Test builder with provider and model."""
        from dspy_examples.builders import BatchBuilder

        # Note: BatchConfig has default provider [{"name": "ollama"}]
        # with_provider appends to the existing list
        builder = BatchBuilder().with_provider("openai", model="gpt-4")

        assert len(builder._config.providers) == 2
        # First provider is the default
        assert builder._config.providers[0] == {"name": "ollama"}
        # Second provider is the one we added
        assert builder._config.providers[1] == {"name": "openai", "model": "gpt-4"}

    def test_builder_with_provider_no_model(self):
        """Test builder with provider without model."""
        from dspy_examples.builders import BatchBuilder

        # Note: BatchConfig has default provider [{"name": "ollama"}]
        builder = BatchBuilder().with_provider("openai")

        assert len(builder._config.providers) == 2
        # First provider is the default
        assert builder._config.providers[0] == {"name": "ollama"}
        # Second provider is the one we added
        assert builder._config.providers[1] == {"name": "openai"}

    def test_builder_with_provider_replaces_default(self):
        """Test that with_providers() replaces the default provider."""
        from dspy_examples.builders import BatchBuilder

        # with_providers replaces the entire list
        builder = BatchBuilder().with_providers(["openai"])

        assert len(builder._config.providers) == 1
        assert builder._config.providers[0] == {"name": "openai"}

    def test_builder_observer_not_yet_attached(self):
        """Test that observers are collected but not attached (documented limitation).

        This test documents that BatchCommand does not yet support observers.
        When BatchCommand gains observer support, this test should be updated.
        """
        from dspy_examples.builders import BatchBuilder
        from dspy_examples.observers import LoggingObserver

        observer = LoggingObserver()
        builder = (BatchBuilder()
            .with_prompts(["test.md"])
            .with_observer(observer))

        batch = builder.build()

        # Observer was collected by builder
        assert observer in builder._observers
        # But BatchCommand does not yet have observer support
        # (this documents the current limitation)
        assert not hasattr(batch, "_observers") or observer not in getattr(batch, "_observers", [])