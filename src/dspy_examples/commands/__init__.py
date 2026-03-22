"""Command pattern for batch processing.

Supports:
- Queue-based execution with persistence
- Multiple prompts, providers, variable sets
- Result aggregation and reporting
"""

from dspy_examples.commands.base import Command, CommandResult
from dspy_examples.commands.nodes import LoadPromptNode, OptimizeNode, SaveResultNode

__all__ = [
    "Command",
    "CommandResult",
    "LoadPromptNode",
    "OptimizeNode",
    "SaveResultNode",
]
