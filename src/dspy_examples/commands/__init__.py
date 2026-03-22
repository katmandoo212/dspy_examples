"""Command pattern for batch processing.

Supports:
- Queue-based execution with persistence
- Multiple prompts, providers, variable sets
- Result aggregation and reporting
"""

from dspy_examples.commands.base import Command, CommandResult

__all__ = ["Command", "CommandResult"]
