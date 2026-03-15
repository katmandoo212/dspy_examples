#!/usr/bin/env python3
"""Main entry point for DSPy optimization."""

from pathlib import Path

from dspy_examples import get_settings
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig


def main() -> None:
    """Run prompt optimization with configured provider and optimizer."""
    settings = get_settings()

    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Optimizer: {settings.optimizer}")

    # Create pipeline with default configuration
    pipeline = OptimizationPipeline(PipelineConfig(
        input_path=Path("prompts/unoptimized_prompt.md"),
        output_path=Path("prompts/optimized_prompt.md"),
    ))

    # Run optimization
    result = pipeline.run()

    print(f"Original length: {result.original_length}")
    print(f"Optimized length: {result.optimized_length}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print(f"Optimizer: {result.optimizer_name}")
    print(f"Provider: {result.provider_name}")

    if result.cache_key:
        print(f"Cache key: {result.cache_key}")


if __name__ == "__main__":
    main()