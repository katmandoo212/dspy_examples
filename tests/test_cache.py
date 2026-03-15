"""Tests for OptimizationCache."""

import json
import tempfile
from pathlib import Path

import pytest


def test_cache_set_and_get():
    """Test setting and getting cached results."""
    from dspy_examples.cache import OptimizationCache
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizationCache(cache_dir=Path(tmpdir))

        result = OptimizationResult(
            optimized_prompt="Optimized",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=100,
            optimized_length=150,
        )

        cache.set("test-key", result)
        cached = cache.get("test-key")

        assert cached is not None
        assert cached.optimized_prompt == "Optimized"
        assert cached.optimizer_name == "bootstrap_fewshot"


def test_cache_get_nonexistent():
    """Test getting nonexistent cache key."""
    from dspy_examples.cache import OptimizationCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizationCache(cache_dir=Path(tmpdir))
        cached = cache.get("nonexistent-key")

        assert cached is None


def test_cache_list_all():
    """Test listing all cached results."""
    from dspy_examples.cache import OptimizationCache
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizationCache(cache_dir=Path(tmpdir))

        result1 = OptimizationResult(
            optimized_prompt="First",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=100,
            optimized_length=150,
        )
        result2 = OptimizationResult(
            optimized_prompt="Second",
            optimizer_name="bootstrap_fewshot",
            provider_name="openai",
            original_length=200,
            optimized_length=250,
        )

        cache.set("key1", result1)
        cache.set("key2", result2)

        all_cached = cache.list_all()

        assert len(all_cached) == 2
        cache_keys = {c["cache_key"] for c in all_cached}
        assert "key1" in cache_keys
        assert "key2" in cache_keys


def test_cache_clear():
    """Test clearing cache."""
    from dspy_examples.cache import OptimizationCache
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizationCache(cache_dir=Path(tmpdir))

        result = OptimizationResult(
            optimized_prompt="Test",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=100,
            optimized_length=150,
        )

        cache.set("test-key", result)
        assert cache.get("test-key") is not None

        cache.clear()
        assert cache.get("test-key") is None


def test_cache_compare():
    """Test comparing cached results."""
    from dspy_examples.cache import OptimizationCache
    from dspy_examples.optimizers.base import OptimizationResult

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizationCache(cache_dir=Path(tmpdir))

        result1 = OptimizationResult(
            optimized_prompt="First",
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            original_length=100,
            optimized_length=150,
        )
        result2 = OptimizationResult(
            optimized_prompt="Second",
            optimizer_name="bootstrap_fewshot",
            provider_name="openai",
            original_length=200,
            optimized_length=250,
        )

        cache.set("key1", result1)
        cache.set("key2", result2)

        compared = cache.compare("key1", "key2")

        assert len(compared) == 2
        assert compared[0].optimized_prompt == "First"
        assert compared[1].optimized_prompt == "Second"