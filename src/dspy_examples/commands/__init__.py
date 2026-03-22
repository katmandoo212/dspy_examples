"""Command pattern for batch processing.

Supports:
- Queue-based execution with persistence
- Multiple prompts, providers, variable sets
- Result aggregation and reporting
"""

from dspy_examples.commands.base import Command, CommandResult
from dspy_examples.commands.nodes import LoadPromptNode, OptimizeNode, SaveResultNode
from dspy_examples.commands.queue import CommandQueue
from dspy_examples.commands.flows import BatchConfig, BatchFlow
from dspy_examples.commands.results import BatchResult, ResultsAggregator
from dspy_examples.commands.batch import BatchCommand, OptimizeCommand

__all__ = [
    "Command",
    "CommandResult",
    "LoadPromptNode",
    "OptimizeNode",
    "SaveResultNode",
    "CommandQueue",
    "BatchConfig",
    "BatchFlow",
    "BatchResult",
    "ResultsAggregator",
    "BatchCommand",
    "OptimizeCommand",
]
