"""Tests for OptimizationPipeline."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_pipeline_config_defaults():
    """Test PipelineConfig has correct defaults."""
    from dspy_examples.pipeline import PipelineConfig

    config = PipelineConfig()

    assert config.provider_name is None
    assert config.optimizer_name is None
    assert config.input_path == Path("prompts/unoptimized_prompt.md")
    assert config.output_path == Path("prompts/optimized_prompt.md")
    assert config.use_cache is True
    assert config.trainset is None


def test_pipeline_get_cache_key():
    """Test cache key generation."""
    from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

    pipeline = OptimizationPipeline(PipelineConfig(use_cache=False))

    key1 = pipeline._get_cache_key("prompt1", "bootstrap_fewshot", "ollama")
    key2 = pipeline._get_cache_key("prompt1", "bootstrap_fewshot", "ollama")
    key3 = pipeline._get_cache_key("prompt2", "bootstrap_fewshot", "ollama")

    assert key1 == key2  # Same inputs = same key
    assert key1 != key3   # Different inputs = different key
    assert len(key1) == 16  # SHA256 truncated to 16 chars


def test_pipeline_load_prompt():
    """Test loading prompt from file."""
    from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test_prompt.md"
        input_file.write_text("Test prompt content")

        config = PipelineConfig(
            input_path=input_file,
            use_cache=False,
        )
        pipeline = OptimizationPipeline(config)
        prompt = pipeline._load_prompt()

        assert prompt == "Test prompt content"


def test_pipeline_save_result():
    """Test saving result to file."""
    from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "optimized.md"

        config = PipelineConfig(
            output_path=output_file,
            use_cache=False,
        )
        pipeline = OptimizationPipeline(config)

        result = OptimizationResult(
            optimized_prompt="Optimized content",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=20,
            optimized_length=18,
        )

        pipeline._save_result(result)

        assert output_file.exists()
        assert output_file.read_text() == "Optimized content"


def test_pipeline_save_result_with_version():
    """Test saving result with versioning when file exists."""
    from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "optimized.md"
        output_file.write_text("Existing content")

        config = PipelineConfig(
            output_path=output_file,
            use_cache=False,
        )
        pipeline = OptimizationPipeline(config)

        result = OptimizationResult(
            optimized_prompt="New content",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=20,
            optimized_length=10,
        )

        pipeline._save_result(result)

        # Should create versioned file
        versioned_file = Path(tmpdir) / "optimized_01.md"
        assert versioned_file.exists()
        assert versioned_file.read_text() == "New content"