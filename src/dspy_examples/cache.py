"""Memento pattern for caching optimization results."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dspy_examples.optimizers.base import OptimizationResult


class OptimizationCache:
    """Memento pattern for caching optimization results."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to settings.cache_dir.
        """
        if cache_dir is None:
            from dspy_examples.settings import get_settings
            settings = get_settings()
            cache_dir = Path(settings.cache_dir)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> OptimizationResult | None:
        """Retrieve cached result if it exists.

        Args:
            cache_key: The unique cache key.

        Returns:
            The cached result, or None if not found.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            # Filter out cached_at which is not part of OptimizationResult
            result_fields = {
                "optimized_prompt",
                "optimizer_name",
                "provider_name",
                "original_length",
                "optimized_length",
                "optimization_time",
                "metadata",
                "cache_key",
            }
            filtered_data = {k: v for k, v in data.items() if k in result_fields}
            return OptimizationResult(**filtered_data)
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, cache_key: str, result: OptimizationResult) -> None:
        """Save optimization result to cache.

        Args:
            cache_key: The unique cache key.
            result: The optimization result to cache.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        data = asdict(result)
        data["cached_at"] = datetime.now().isoformat()

        cache_file.write_text(json.dumps(data, indent=2, default=str))

    def list_all(self) -> list[dict[str, Any]]:
        """List all cached optimizations with metadata.

        Returns:
            List of cache entries with metadata.
        """
        results = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                results.append({
                    "cache_key": cache_file.stem,
                    "optimizer_name": data.get("optimizer_name"),
                    "provider_name": data.get("provider_name"),
                    "cached_at": data.get("cached_at"),
                    "original_length": data.get("original_length"),
                    "optimized_length": data.get("optimized_length"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def compare(self, *cache_keys: str) -> list[OptimizationResult]:
        """Compare multiple cached optimizations.

        Args:
            *cache_keys: Cache keys to compare.

        Returns:
            List of cached results.
        """
        results = []
        for key in cache_keys:
            if result := self.get(key):
                results.append(result)
        return results