"""Core PocketFlow implementation.

A minimal (~100 line) implementation of Node/Flow pattern for AI workflows.
Based on PocketFlow architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Node(ABC):
    """Base class for workflow nodes.

    Each node has three phases:
    - prep: Prepare data from shared state
    - exec: Execute the node's logic
    - post: Update shared state with results

    Returns an action name for flow routing.
    """

    def __init__(self) -> None:
        self.successors: dict[str, Node] = {}

    def __rshift__(self, other: Node) -> Node:
        """Connect this node to another: node_a >> node_b."""
        self.successors["default"] = other
        return other

    def prep(self, shared: dict[str, Any]) -> Any:
        """Prepare data from shared state. Override in subclasses."""
        return None

    @abstractmethod
    def exec(self, prep_res: Any) -> Any:
        """Execute the node's logic. Must be implemented by subclasses."""
        pass

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> str | None:
        """Update shared state with results. Override in subclasses.

        Returns:
            Action name for routing, or None for default.
        """
        return None

    def run(self, shared: dict[str, Any]) -> str | None:
        """Execute this node and return action name."""
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        action = self.post(shared, prep_res, exec_res)
        return action

    def set_successor(self, node: Node, action: str = "default") -> Node:
        """Set a successor node for a specific action."""
        self.successors[action] = node
        return node


class Flow(Node):
    """A flow that orchestrates multiple nodes in sequence."""

    def __init__(self, start: Node | None = None) -> None:
        super().__init__()
        self.start = start

    def exec(self, prep_res: Any) -> Any:
        """Execute the flow by running nodes in sequence."""
        if self.start is None:
            return None

        current = self.start
        shared = prep_res if prep_res is not None else {}

        while current is not None:
            action = current.run(shared)

            # Get next node based on action
            if action and action in current.successors:
                current = current.successors[action]
            elif "default" in current.successors:
                current = current.successors["default"]
            else:
                current = None

        return shared

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Pass shared state to exec."""
        return shared

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: Any
    ) -> str | None:
        """Flow doesn't return an action, just ends."""
        return None


class BatchNode(Node):
    """A node that processes items in batch.

    Subclasses implement:
    - prep: Return list of items to process
    - exec: Process a single item
    - post: Aggregate results
    """

    def prep(self, shared: dict[str, Any]) -> list[Any]:
        """Return list of items to process."""
        return []

    @abstractmethod
    def exec(self, item: Any) -> Any:
        """Process a single item. Must be implemented by subclasses."""
        pass

    def post(
        self, shared: dict[str, Any], prep_res: list[Any], exec_res: list[Any]
    ) -> str | None:
        """Aggregate batch results."""
        shared["batch_results"] = exec_res
        return None

    def run(self, shared: dict[str, Any]) -> str | None:
        """Execute batch processing."""
        items = self.prep(shared)
        results = []
        for item in items:
            result = self.exec(item)
            results.append(result)
        action = self.post(shared, items, results)
        return action


class BatchFlow(Flow):
    """A flow that processes multiple configurations in batch."""

    def __init__(self, start: Node | None = None) -> None:
        super().__init__(start)
        self.batch_configs: list[dict[str, Any]] = []

    def set_batch_configs(self, configs: list[dict[str, Any]]) -> BatchFlow:
        """Set batch configurations to process."""
        self.batch_configs = configs
        return self

    def exec(self, prep_res: Any) -> list[dict[str, Any]]:
        """Execute flow for each batch configuration."""
        results = []

        for config in self.batch_configs:
            # Create fresh shared state for each batch
            shared = dict(prep_res) if prep_res else {}
            shared.update(config)

            if self.start:
                current = self.start
                while current is not None:
                    action = current.run(shared)
                    if action and action in current.successors:
                        current = current.successors[action]
                    elif "default" in current.successors:
                        current = current.successors["default"]
                    else:
                        current = None

            results.append(shared)

        return results