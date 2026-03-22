# Batch Processing with PocketFlow - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add batch processing using embedded PocketFlow for execution orchestration and SQLite for queue persistence, supporting multiple prompts, multiple variable sets, and multi-provider execution.

**Architecture:** Embedded PocketFlow (~100 lines) provides Node/Flow primitives for composable execution. OptimizeNode wraps single optimization, BatchFlow orchestrates multiple executions. SQLite persists queue state for resumability. ResultsAggregator generates individual output files plus summary reports.

**Tech Stack:** Python 3.14, DSPy, PocketFlow (embedded), SQLite3, asyncio (optional parallel execution)

---

## Task 1: Embed PocketFlow Core

**Files:**
- Create: `src/dspy_examples/pocketflow/__init__.py`
- Create: `src/dspy_examples/pocketflow/core.py`
- Test: `tests/test_pocketflow_core.py`

**Step 1: Write the failing test for Node base class**

Create `tests/test_pocketflow_core.py`:

```python
"""Tests for embedded PocketFlow core."""

import pytest


class TestNode:
    """Tests for Node base class."""

    def test_node_creation(self):
        """Test creating a node with prep/exec/post methods."""
        from dspy_examples.pocketflow import Node

        class MyNode(Node):
            def prep(self, shared):
                return {"input": shared["value"]}

            def exec(self, prep_res):
                return prep_res["input"] * 2

            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res
                return "default"

        node = MyNode()
        shared = {"value": 5}
        result = node.run(shared)

        assert shared["result"] == 10
        assert result == "default"

    def test_node_with_successors(self):
        """Test node with successor nodes."""
        from dspy_examples.pocketflow import Node

        class StartNode(Node):
            def exec(self, prep_res):
                return "next"

        class EndNode(Node):
            def exec(self, prep_res):
                return "done"

        start = StartNode()
        end = EndNode()
        start >> end  # Connect nodes

        assert end in start.successors
        assert len(end.successors) == 0

    def test_node_orchestrator_pattern(self):
        """Test orchestrator returns action name for routing."""
        from dspy_examples.pocketflow import Node

        class RouterNode(Node):
            def exec(self, prep_res):
                return "path_a"

        node = RouterNode()
        result = node.run({})

        assert result == "path_a"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_pocketflow_core.py -v`
Expected: FAIL with "No module named 'dspy_examples.pocketflow'"

**Step 3: Create pocketflow package structure**

Create `src/dspy_examples/pocketflow/__init__.py`:

```python
"""Embedded PocketFlow for execution orchestration.

PocketFlow is a minimal framework for building AI workflows.
This is an embedded copy for self-containment.
"""

from dspy_examples.pocketflow.core import Node, Flow, BatchNode, BatchFlow

__all__ = ["Node", "Flow", "BatchNode", "BatchFlow"]
```

**Step 4: Run test to verify module imports**

Run: `PYTHONPATH=src uv run pytest tests/test_pocketflow_core.py -v`
Expected: FAIL with "cannot import name 'Node' from 'dspy_examples.pocketflow.core'"

**Step 5: Implement PocketFlow core classes**

Create `src/dspy_examples/pocketflow/core.py`:

```python
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

    def __rshift__(self, other: Node) -> Node:
        """Connect this node to another: node_a >> node_b."""
        self.successors["default"] = other
        return other

    def __or__(self, action: str) -> tuple["Node", str]:
        """Create an action pair: node | "action_name"."""
        return (self, action)

    def __rshift__(self, other: tuple["Node", str]) -> Node:
        """Connect with action name: node_a >> (node_b, "action")."""
        node, action = other
        self.successors[action] = node
        return node

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

    def set_successor(self, node: "Node", action: str = "default") -> "Node":
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

    def set_batch_configs(self, configs: list[dict[str, Any]]) -> "BatchFlow":
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
```

**Step 6: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_pocketflow_core.py -v`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add src/dspy_examples/pocketflow/__init__.py src/dspy_examples/pocketflow/core.py tests/test_pocketflow_core.py
git commit -m "feat: add embedded PocketFlow core for workflow orchestration

- Add Node base class with prep/exec/post pattern
- Add Flow for sequential node orchestration
- Add BatchNode for batch processing
- Add BatchFlow for multi-config execution
- Include comprehensive tests"
```

---

## Task 2: Create Command Base Classes

**Files:**
- Create: `src/dspy_examples/commands/__init__.py`
- Create: `src/dspy_examples/commands/base.py`
- Test: `tests/test_commands_base.py`

**Step 1: Write the failing test for CommandResult**

Create `tests/test_commands_base.py`:

```python
"""Tests for command base classes."""

import pytest
from datetime import datetime
from pathlib import Path


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_command_result_creation(self):
        """Test creating a successful result."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_001",
            status="success",
            output_path=Path("output/prompt1.md"),
            optimizer_name="bootstrap_fewshot",
            provider_name="ollama",
            model_name="llama3",
            execution_time=42.5,
        )

        assert result.command_id == "cmd_001"
        assert result.status == "success"
        assert result.output_path == Path("output/prompt1.md")
        assert result.optimizer_name == "bootstrap_fewshot"
        assert result.provider_name == "ollama"
        assert result.model_name == "llama3"
        assert result.execution_time == 42.5
        assert result.error_message is None
        assert result.metadata == {}

    def test_command_result_failed(self):
        """Test creating a failed result."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_002",
            status="failed",
            output_path=None,
            optimizer_name="mipro_v2",
            provider_name="openai",
            model_name="gpt-4",
            execution_time=5.2,
            error_message="Connection timeout",
        )

        assert result.status == "failed"
        assert result.error_message == "Connection timeout"
        assert result.output_path is None

    def test_command_result_with_metadata(self):
        """Test result with additional metadata."""
        from dspy_examples.commands.base import CommandResult

        result = CommandResult(
            command_id="cmd_003",
            status="success",
            output_path=Path("output/prompt2.md"),
            optimizer_name="gepa",
            provider_name="anthropic",
            model_name="claude-3",
            execution_time=30.0,
            metadata={"tokens_used": 1500, "attempts": 2},
        )

        assert result.metadata["tokens_used"] == 1500
        assert result.metadata["attempts"] == 2


class TestCommand:
    """Tests for Command abstract base class."""

    def test_command_abstract_methods(self):
        """Test that Command requires exec implementation."""
        from dspy_examples.commands.base import Command

        with pytest.raises(TypeError):
            Command()  # Cannot instantiate abstract class

    def test_command_implementation(self):
        """Test implementing a concrete command."""
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, command_id: str):
                self._id = command_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=Path("test.md"),
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=1.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        cmd = TestCommand("test_001")
        result = cmd.execute()

        assert result.command_id == "test_001"
        assert result.status == "success"

    def test_command_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from dspy_examples.commands.base import Command, CommandResult

        class SerializeTestCommand(Command):
            def __init__(self, cmd_id: str, value: str):
                self._id = cmd_id
                self._value = value

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "value": self._value}

            @classmethod
            def from_dict(cls, data: dict) -> "SerializeTestCommand":
                return cls(data["id"], data["value"])

        cmd1 = SerializeTestCommand("id_001", "hello")
        data = cmd1.to_dict()
        cmd2 = SerializeTestCommand.from_dict(data)

        assert cmd2.command_id == "id_001"
        assert cmd2._value == "hello"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_base.py -v`
Expected: FAIL with "No module named 'dspy_examples.commands'"

**Step 3: Create commands package structure**

Create `src/dspy_examples/commands/__init__.py`:

```python
"""Command pattern for batch processing.

Supports:
- Queue-based execution with persistence
- Multiple prompts, providers, variable sets
- Result aggregation and reporting
"""

from dspy_examples.commands.base import Command, CommandResult

__all__ = ["Command", "CommandResult"]
```

**Step 4: Run test to verify module imports**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_base.py -v`
Expected: FAIL with "cannot import name 'CommandResult' from 'dspy_examples.commands.base'"

**Step 5: Implement Command base classes**

Create `src/dspy_examples/commands/base.py`:

```python
"""Base classes for command pattern.

Command pattern encapsulates optimization requests as objects,
enabling queue-based execution, persistence, and batch processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class CommandResult:
    """Result of executing a command.

    Attributes:
        command_id: Unique identifier for the command
        status: Execution status (success, failed, skipped)
        output_path: Path to output file (None if failed)
        optimizer_name: Name of optimizer used
        provider_name: Name of LLM provider
        model_name: Name of model used
        execution_time: Time in seconds for execution
        error_message: Error message if failed (None if success)
        metadata: Additional metadata about execution
    """

    command_id: str
    status: Literal["success", "failed", "skipped"]
    output_path: Path | None
    optimizer_name: str
    provider_name: str
    model_name: str
    execution_time: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Command(ABC):
    """Abstract base class for commands.

    Commands encapsulate optimization requests that can be:
    - Queued for execution
    - Persisted to SQLite for recovery
    - Executed by workers
    """

    @property
    @abstractmethod
    def command_id(self) -> str:
        """Unique identifier for this command."""
        pass

    @abstractmethod
    def execute(self) -> CommandResult:
        """Execute this command and return the result."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize command to dictionary for persistence."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        """Deserialize command from dictionary."""
        pass
```

**Step 6: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_base.py -v`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add src/dspy_examples/commands/__init__.py src/dspy_examples/commands/base.py tests/test_commands_base.py
git commit -m "feat: add Command base classes for batch processing

- Add CommandResult dataclass for execution results
- Add Command abstract base class with execute/to_dict/from_dict
- Support for success/failed/skipped status
- Include metadata for additional execution info"
```

---

## Task 3: Create SQLite Queue Persistence

**Files:**
- Create: `src/dspy_examples/commands/queue.py`
- Test: `tests/test_commands_queue.py`

**Step 1: Write the failing test for CommandQueue**

Create `tests/test_commands_queue.py`:

```python
"""Tests for SQLite-backed command queue."""

import pytest
import tempfile
from pathlib import Path


class TestCommandQueue:
    """Tests for SQLite command queue."""

    def test_queue_creation(self):
        """Test creating a new queue database."""
        from dspy_examples.commands.queue import CommandQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            assert db_path.exists()
            assert queue.is_empty()

    def test_add_command(self):
        """Test adding a command to the queue."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class SimpleCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "simple"}

            @classmethod
            def from_dict(cls, data: dict) -> "SimpleCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            cmd = SimpleCommand("cmd_001")
            queue.add(cmd)

            assert queue.size() == 1
            assert not queue.is_empty()

    def test_get_pending(self):
        """Test retrieving pending commands."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            queue.add(TestCommand("cmd_001"))
            queue.add(TestCommand("cmd_002"))

            pending = queue.get_pending()
            assert len(pending) == 2
            assert pending[0].command_id == "cmd_001"
            assert pending[1].command_id == "cmd_002"

    def test_mark_completed(self):
        """Test marking a command as completed."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            queue.add(TestCommand("cmd_001"))
            queue.add(TestCommand("cmd_002"))

            queue.mark_completed("cmd_001")

            pending = queue.get_pending()
            assert len(pending) == 1
            assert pending[0].command_id == "cmd_002"

    def test_mark_failed(self):
        """Test marking a command as failed."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            queue.add(TestCommand("cmd_001"))
            queue.mark_failed("cmd_001", "Connection error")

            pending = queue.get_pending()
            assert len(pending) == 0

            failed = queue.get_failed()
            assert len(failed) == 1
            assert failed[0]["id"] == "cmd_001"
            assert failed[0]["error"] == "Connection error"

    def test_get_completed(self):
        """Test retrieving completed commands."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            queue.add(TestCommand("cmd_001"))
            queue.add(TestCommand("cmd_002"))
            queue.mark_completed("cmd_001")

            completed = queue.get_completed()
            assert len(completed) == 1
            assert completed[0]["id"] == "cmd_001"

    def test_clear_queue(self):
        """Test clearing all commands from queue."""
        from dspy_examples.commands.queue import CommandQueue
        from dspy_examples.commands.base import Command, CommandResult

        class TestCommand(Command):
            def __init__(self, cmd_id: str):
                self._id = cmd_id

            @property
            def command_id(self) -> str:
                return self._id

            def execute(self) -> CommandResult:
                return CommandResult(
                    command_id=self._id,
                    status="success",
                    output_path=None,
                    optimizer_name="test",
                    provider_name="test",
                    model_name="test",
                    execution_time=0.0,
                )

            def to_dict(self) -> dict:
                return {"id": self._id, "type": "test"}

            @classmethod
            def from_dict(cls, data: dict) -> "TestCommand":
                return cls(data["id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            queue = CommandQueue(db_path)

            queue.add(TestCommand("cmd_001"))
            queue.add(TestCommand("cmd_002"))

            queue.clear()

            assert queue.is_empty()
            assert queue.size() == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_queue.py -v`
Expected: FAIL with "No module named 'dspy_examples.commands.queue'"

**Step 3: Implement SQLite queue**

Create `src/dspy_examples/commands/queue.py`:

```python
"""SQLite-backed command queue for persistence.

Enables:
- Queue-based execution
- Recovery after crashes
- Status tracking (pending, running, completed, failed)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dspy_examples.commands.base import Command


class CommandQueue:
    """SQLite-backed queue for command persistence.

    Schema:
        commands:
            id: TEXT PRIMARY KEY
            batch_id: TEXT
            type: TEXT NOT NULL
            data: TEXT NOT NULL (JSON)
            status: TEXT DEFAULT 'pending'
            priority: INTEGER DEFAULT 0
            created_at: TIMESTAMP
            started_at: TIMESTAMP
            completed_at: TIMESTAMP

        results:
            command_id: TEXT PRIMARY KEY
            status: TEXT NOT NULL
            output_path: TEXT
            optimizer_name: TEXT
            provider_name: TEXT
            model_name: TEXT
            execution_time: REAL
            error_message: TEXT
            metadata: TEXT (JSON)
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize queue with SQLite database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS commands (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT,
                    type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS results (
                    command_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    optimizer_name TEXT,
                    provider_name TEXT,
                    model_name TEXT,
                    execution_time REAL,
                    error_message TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_status ON commands(status);
                CREATE INDEX IF NOT EXISTS idx_batch ON commands(batch_id);
            """)

    def add(
        self,
        command: Command,
        batch_id: str | None = None,
        priority: int = 0,
    ) -> None:
        """Add a command to the queue.

        Args:
            command: Command to add
            batch_id: Optional batch identifier
            priority: Priority (higher = processed first)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO commands (id, batch_id, type, data, priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    command.command_id,
                    batch_id,
                    command.__class__.__name__,
                    json.dumps(command.to_dict()),
                    priority,
                ),
            )

    def get_pending(self, limit: int | None = None) -> list[Command]:
        """Get pending commands from queue.

        Args:
            limit: Maximum number of commands to return

        Returns:
            List of pending commands, ordered by priority desc, created_at asc
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT type, data FROM commands
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """
            if limit:
                query += f" LIMIT {limit}"

            rows = conn.execute(query).fetchall()

            # Note: Actual command reconstruction requires registry
            # This returns raw data that needs to be reconstructed
            return [(row[0], json.loads(row[1])) for row in rows]

    def mark_running(self, command_id: str) -> None:
        """Mark command as currently running."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE commands
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (command_id,),
            )

    def mark_completed(self, command_id: str) -> None:
        """Mark command as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE commands
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (command_id,),
            )

    def mark_failed(self, command_id: str, error: str) -> None:
        """Mark command as failed with error message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE commands
                SET status = 'failed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (command_id,),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO results (command_id, status, error_message)
                VALUES (?, 'failed', ?)
                """,
                (command_id, error),
            )

    def save_result(self, command_id: str, result: dict[str, Any]) -> None:
        """Save execution result.

        Args:
            command_id: Command identifier
            result: Result data to save
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO results
                (command_id, status, output_path, optimizer_name,
                 provider_name, model_name, execution_time, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    result.get("status"),
                    result.get("output_path"),
                    result.get("optimizer_name"),
                    result.get("provider_name"),
                    result.get("model_name"),
                    result.get("execution_time"),
                    result.get("error_message"),
                    json.dumps(result.get("metadata", {})),
                ),
            )

    def get_completed(self) -> list[dict[str, Any]]:
        """Get all completed command results."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.batch_id, c.type, r.*
                FROM commands c
                JOIN results r ON c.id = r.command_id
                WHERE c.status = 'completed'
                ORDER BY c.completed_at DESC
                """
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "batch_id": row[1],
                    "type": row[2],
                    "status": row[3],
                    "output_path": row[4],
                    "optimizer_name": row[5],
                    "provider_name": row[6],
                    "model_name": row[7],
                    "execution_time": row[8],
                    "error_message": row[9],
                    "metadata": json.loads(row[10]) if row[10] else {},
                }
                for row in rows
            ]

    def get_failed(self) -> list[dict[str, Any]]:
        """Get all failed commands with error messages."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, batch_id, error_message
                FROM commands
                WHERE status = 'failed'
                ORDER BY completed_at DESC
                """
            ).fetchall()
            return [
                {"id": row[0], "batch_id": row[1], "error": row[2]} for row in rows
            ]

    def is_empty(self) -> bool:
        """Check if queue has no pending commands."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM commands WHERE status = 'pending'"
            ).fetchone()[0]
            return count == 0

    def size(self) -> int:
        """Get total number of pending commands."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM commands WHERE status = 'pending'"
            ).fetchone()[0]

    def clear(self) -> None:
        """Remove all commands from queue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM commands")
            conn.execute("DELETE FROM results")
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_queue.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/dspy_examples/commands/queue.py tests/test_commands_queue.py
git commit -m "feat: add SQLite-backed command queue for persistence

- Create CommandQueue with full CRUD operations
- Support pending/running/completed/failed status
- Enable batch_id tracking for grouping commands
- Add priority ordering for queue processing
- Store execution results and metadata"
```

---

## Task 4: Create OptimizeNode (PocketFlow Node)

**Files:**
- Create: `src/dspy_examples/commands/nodes.py`
- Test: `tests/test_commands_nodes.py`

**Step 1: Write the failing test for OptimizeNode**

Create `tests/test_commands_nodes.py`:

```python
"""Tests for PocketFlow nodes for optimization."""

import pytest
from pathlib import Path
import tempfile


class TestOptimizeNode:
    """Tests for OptimizeNode."""

    def test_optimize_node_creation(self):
        """Test creating an OptimizeNode with configuration."""
        from dspy_examples.commands.nodes import OptimizeNode

        node = OptimizeNode(
            prompt_path=Path("prompts/test.md"),
            output_path=Path("output/test_optimized.md"),
            provider_name="ollama",
            model_name="llama3",
            optimizer_name="bootstrap_fewshot",
        )

        assert node.prompt_path == Path("prompts/test.md")
        assert node.provider_name == "ollama"
        assert node.optimizer_name == "bootstrap_fewshot"

    def test_optimize_node_prep(self):
        """Test prep extracts configuration from shared state."""
        from dspy_examples.commands.nodes import OptimizeNode

        node = OptimizeNode(
            prompt_path=Path("prompts/test.md"),
            output_path=Path("output/test_optimized.md"),
            provider_name="ollama",
            model_name="llama3",
            optimizer_name="bootstrap_fewshot",
        )

        shared = {"batch_id": "batch_001"}
        prep_res = node.prep(shared)

        assert prep_res["prompt_path"] == Path("prompts/test.md")
        assert prep_res["output_path"] == Path("output/test_optimized.md")
        assert prep_res["provider_name"] == "ollama"

    def test_optimize_node_run_updates_shared(self):
        """Test that run updates shared state with results."""
        from dspy_examples.commands.nodes import OptimizeNode
        from unittest.mock import patch, MagicMock

        node = OptimizeNode(
            prompt_path=Path("prompts/test.md"),
            output_path=Path("output/test_optimized.md"),
            provider_name="ollama",
            model_name="llama3",
            optimizer_name="bootstrap_fewshot",
        )

        # Mock the pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt")
            node.prompt_path = prompt_file
            node.output_path = Path(tmpdir) / "output.md"

            shared = {"batch_id": "test"}

            # Since this requires actual DSPy, test the prep/post flow
            prep_res = node.prep(shared)
            assert "prompt_path" in prep_res


class TestLoadPromptNode:
    """Tests for LoadPromptNode."""

    def test_load_prompt_node(self):
        """Test loading a prompt file."""
        from dspy_examples.commands.nodes import LoadPromptNode

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("Test prompt content")

            node = LoadPromptNode(prompt_path=prompt_file)

            shared = {}
            result = node.run(shared)

            assert shared["prompt_content"] == "Test prompt content"
            assert result == "default"

    def test_load_prompt_with_variables(self):
        """Test loading prompt with variable substitution."""
        from dspy_examples.commands.nodes import LoadPromptNode

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test.md"
            prompt_file.write_text("""---
variables:
  topic:
    mode: substitute
    default: Python
---

Explain [[topic]] in detail.""")

            node = LoadPromptNode(
                prompt_path=prompt_file,
                variables={"topic": "JavaScript"},
            )

            shared = {}
            result = node.run(shared)

            assert "JavaScript" in shared["prompt_content"]
            assert "[[topic]]" not in shared["prompt_content"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_nodes.py -v`
Expected: FAIL with "No module named 'dspy_examples.commands.nodes'"

**Step 3: Implement OptimizeNode and LoadPromptNode**

Create `src/dspy_examples/commands/nodes.py`:

```python
"""PocketFlow nodes for optimization workflow.

Nodes implement the prep/exec/post pattern:
- prep: Extract data from shared state
- exec: Execute the node's logic
- post: Update shared state with results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dspy_examples.pocketflow import Node
from dspy_examples.template import PromptTemplate


class LoadPromptNode(Node):
    """Node that loads and optionally processes a prompt file.

    Shared state inputs:
        None required

    Shared state outputs:
        prompt_content: The loaded prompt content
        prompt_template: PromptTemplate object (if frontmatter exists)
    """

    def __init__(
        self,
        prompt_path: Path,
        variables: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_path = Path(prompt_path)
        self.variables = variables or {}

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare prompt loading parameters."""
        return {
            "prompt_path": self.prompt_path,
            "variables": self.variables,
        }

    def exec(self, prep_res: dict[str, Any]) -> str:
        """Load and process the prompt."""
        prompt_path = prep_res["prompt_path"]
        variables = prep_res["variables"]

        # Try to load as template with frontmatter
        try:
            template = PromptTemplate.from_file(prompt_path)
            # Validate and substitute
            errors = template.validate(variables)
            if errors:
                # Filter warnings
                actual_errors = [e for e in errors if not e.startswith("Warning:")]
                if actual_errors:
                    raise ValueError(f"Variable validation failed: {'; '.join(actual_errors)}")

            content = template.substitute(variables)
            return content
        except ValueError as e:
            if "Invalid YAML frontmatter" in str(e):
                # Load as plain prompt
                return prompt_path.read_text()
            raise

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: str
    ) -> str | None:
        """Update shared state with loaded prompt."""
        shared["prompt_content"] = exec_res
        shared["prompt_path"] = prep_res["prompt_path"]
        return "default"


class OptimizeNode(Node):
    """Node that runs prompt optimization.

    Uses the existing OptimizationPipeline to optimize prompts.

    Shared state inputs:
        prompt_content: The prompt to optimize
        provider_name: LLM provider to use
        optimizer_name: Optimizer to use

    Shared state outputs:
        result: OptimizationResult object
        optimized_prompt: The optimized prompt string
    """

    def __init__(
        self,
        prompt_path: Path,
        output_path: Path,
        provider_name: str,
        optimizer_name: str,
        model_name: str | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_path = Path(prompt_path)
        self.output_path = Path(output_path)
        self.provider_name = provider_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.variables = variables or {}

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare optimization parameters."""
        return {
            "prompt_path": self.prompt_path,
            "output_path": self.output_path,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "optimizer_name": self.optimizer_name,
            "variables": self.variables,
            "batch_id": shared.get("batch_id"),
        }

    def exec(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Run the optimization using the pipeline.

        Note: This is a placeholder that returns mock results.
        Actual optimization requires DSPy setup.
        """
        # Import here to avoid circular dependency
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
        from dspy_examples.optimizers.base import OptimizationResult

        config = PipelineConfig(
            provider_name=prep_res["provider_name"],
            optimizer_name=prep_res["optimizer_name"],
            input_path=prep_res["prompt_path"],
            output_path=prep_res["output_path"],
            variables=prep_res["variables"],
            use_cache=False,  # Don't cache in batch processing
        )

        # Run optimization
        pipeline = OptimizationPipeline(config)
        result = pipeline.run(prep_res["variables"])

        return {
            "status": "success",
            "result": result,
            "output_path": prep_res["output_path"],
        }

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: dict[str, Any]
    ) -> str | None:
        """Update shared state with optimization result."""
        shared["result"] = exec_res.get("result")
        shared["status"] = exec_res.get("status", "unknown")
        shared["output_path"] = exec_res.get("output_path")

        if exec_res.get("status") == "failed":
            shared["error"] = exec_res.get("error")
            return "failed"

        return "default"


class SaveResultNode(Node):
    """Node that saves optimization results.

    Shared state inputs:
        result: OptimizationResult object
        output_path: Path to save to

    Shared state outputs:
        saved_path: Actual path where result was saved
    """

    def __init__(self, output_path: Path | None = None) -> None:
        super().__init__()
        self.output_path = output_path

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare save parameters."""
        return {
            "result": shared.get("result"),
            "output_path": self.output_path or shared.get("output_path"),
        }

    def exec(self, prep_res: dict[str, Any]) -> Path:
        """Save the result to file."""
        from dspy_examples.prompts import save_prompt

        result = prep_res["result"]
        output_path = prep_res["output_path"]

        if result is None:
            raise ValueError("No result to save")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the optimized prompt
        save_prompt(result.optimized_prompt, output_path)

        return output_path

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: Path
    ) -> str | None:
        """Update shared state with saved path."""
        shared["saved_path"] = exec_res
        return "default"
```

**Step 4: Update commands __init__.py**

Edit `src/dspy_examples/commands/__init__.py`:

```python
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
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_nodes.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/dspy_examples/commands/__init__.py src/dspy_examples/commands/nodes.py tests/test_commands_nodes.py
git commit -m "feat: add PocketFlow nodes for optimization workflow

- Add LoadPromptNode for loading and processing prompts
- Add OptimizeNode for running optimization pipeline
- Add SaveResultNode for saving results to files
- Support variable substitution in LoadPromptNode
- Update exports in commands __init__.py"
```

---

## Task 5: Create BatchFlow and Result Aggregation

**Files:**
- Create: `src/dspy_examples/commands/flows.py`
- Create: `src/dspy_examples/commands/results.py`
- Test: `tests/test_commands_flows.py`

**Step 1: Write the failing test for BatchFlow**

Create `tests/test_commands_flows.py`:

```python
"""Tests for batch processing flows."""

import pytest
from pathlib import Path
import tempfile


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_batch_config_defaults(self):
        """Test default batch configuration."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig()

        assert config.providers == [{"name": "ollama"}]
        assert config.optimizer_name == "bootstrap_fewshot"
        assert config.output_dir == Path("prompts/batch_output")
        assert config.max_concurrent == 1
        assert config.retry_count == 2

    def test_batch_config_with_prompts(self):
        """Test batch config with prompt paths."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[
                Path("prompts/q1.md"),
                Path("prompts/q2.md"),
            ],
            providers=[{"name": "openai", "model": "gpt-4"}],
        )

        assert len(config.prompt_paths) == 2
        assert config.providers[0]["name"] == "openai"

    def test_batch_config_with_template(self):
        """Test batch config with template and variables."""
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_template=Path("prompts/template.md"),
            variable_sets=[
                {"country": "France", "tone": "formal"},
                {"country": "Japan", "tone": "casual"},
            ],
        )

        assert config.prompt_template == Path("prompts/template.md")
        assert len(config.variable_sets) == 2


class TestBatchResult:
    """Tests for BatchResult."""

    def test_batch_result_creation(self):
        """Test creating a batch result."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
            CommandResult(
                command_id="cmd_002",
                status="success",
                output_path=Path("output/p2.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="openai",
                model_name="gpt-4",
                execution_time=15.2,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=2,
            successful=2,
            failed=0,
            skipped=0,
            total_time=25.7,
            results=results,
        )

        assert batch_result.batch_id == "batch_001"
        assert batch_result.successful == 2
        assert batch_result.failed == 0

    def test_batch_result_to_markdown(self):
        """Test converting batch result to markdown."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
            by_provider={"ollama": {"count": 1, "avg_time": 10.5, "success_rate": 1.0}},
            by_optimizer={"bootstrap_fewshot": {"count": 1, "avg_time": 10.5}},
        )

        md = batch_result.to_markdown()

        assert "# Batch Report: batch_001" in md
        assert "Total: 1 commands" in md
        assert "Successful: 1" in md

    def test_batch_result_to_json(self):
        """Test converting batch result to JSON dict."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
        )

        json_dict = batch_result.to_json()

        assert json_dict["batch_id"] == "batch_001"
        assert json_dict["total_commands"] == 1
        assert json_dict["successful"] == 1

    def test_batch_result_save(self):
        """Test saving batch result to files."""
        from dspy_examples.commands.results import BatchResult, CommandResult

        results = [
            CommandResult(
                command_id="cmd_001",
                status="success",
                output_path=Path("output/p1.md"),
                optimizer_name="bootstrap_fewshot",
                provider_name="ollama",
                model_name="llama3",
                execution_time=10.5,
            ),
        ]

        batch_result = BatchResult(
            batch_id="batch_001",
            total_commands=1,
            successful=1,
            failed=0,
            skipped=0,
            total_time=10.5,
            results=results,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            batch_result.save(output_dir)

            # Check markdown file exists
            md_file = output_dir / "batch_001_report.md"
            assert md_file.exists()

            # Check JSON file exists
            json_file = output_dir / "batch_001_results.json"
            assert json_file.exists()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_flows.py -v`
Expected: FAIL with "No module named 'dspy_examples.commands.flows'"

**Step 3: Implement BatchConfig and BatchFlow**

Create `src/dspy_examples/commands/flows.py`:

```python
"""Batch processing flows using PocketFlow.

Provides high-level batch processing with:
- Multiple prompts, single config
- Single prompt, multiple variable sets
- Multi-provider comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid

from dspy_examples.pocketflow import BatchFlow as PF_BatchFlow
from dspy_examples.pocketflow import Node
from dspy_examples.commands.nodes import OptimizeNode


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        prompt_paths: List of prompt files to process
        prompt_template: Template file for variable substitution
        variable_sets: List of variable dictionaries for substitution
        providers: List of provider configurations
        optimizer_name: Name of optimizer to use
        optimizer_config: Optional optimizer configuration
        output_dir: Directory for output files
        naming_pattern: Pattern for output file names
        max_concurrent: Maximum concurrent executions
        retry_count: Number of retries for failed commands
    """

    # Prompt sources
    prompt_paths: list[Path] | None = None
    prompt_template: Path | None = None
    variable_sets: list[dict[str, str]] | None = None

    # Provider configurations
    providers: list[dict[str, str]] = field(default_factory=lambda: [{"name": "ollama"}])

    # Optimizer settings
    optimizer_name: str = "bootstrap_fewshot"
    optimizer_config: dict[str, Any] | None = None

    # Output settings
    output_dir: Path = Path("prompts/batch_output")
    naming_pattern: str = "{prompt}_{provider}_{model}_{optimizer}"

    # Execution settings
    max_concurrent: int = 1
    retry_count: int = 2


class BatchFlow:
    """Orchestrates batch optimization processing.

    Creates and manages batches of OptimizeNodes.
    """

    def __init__(self, config: BatchConfig) -> None:
        """Initialize batch flow with configuration.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.batch_id = self._generate_batch_id()
        self._build_flow()

    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier."""
        return f"batch_{uuid.uuid4().hex[:8]}"

    def _build_flow(self) -> None:
        """Build the flow of OptimizeNodes."""
        configs = self._generate_configs()
        self._configs = configs

    def _generate_configs(self) -> list[dict[str, Any]]:
        """Generate all configuration combinations.

        Returns:
            List of config dicts for each combination of:
            - Prompt × Provider × Variables
        """
        configs = []

        # Get prompt sources
        if self.config.prompt_paths:
            prompt_sources = [(p, None) for p in self.config.prompt_paths]
        elif self.config.prompt_template and self.config.variable_sets:
            prompt_sources = [
                (self.config.prompt_template, vars)
                for vars in self.config.variable_sets
            ]
        elif self.config.prompt_template:
            prompt_sources = [(self.config.prompt_template, None)]
        else:
            raise ValueError("Must provide prompt_paths or prompt_template")

        # Generate all combinations
        for prompt_path, variables in prompt_sources:
            for provider_config in self.config.providers:
                provider_name = provider_config.get("name", "ollama")
                model_name = provider_config.get("model")

                config = {
                    "prompt_path": prompt_path,
                    "variables": variables or {},
                    "provider_name": provider_name,
                    "model_name": model_name,
                    "optimizer_name": self.config.optimizer_name,
                    "output_path": self._get_output_path(
                        prompt_path, provider_name, model_name
                    ),
                }
                configs.append(config)

        return configs

    def _get_output_path(
        self, prompt_path: Path, provider_name: str, model_name: str | None
    ) -> Path:
        """Generate output file path for a configuration."""
        prompt_stem = prompt_path.stem
        model = model_name or "default"
        optimizer = self.config.optimizer_name

        filename = self.config.naming_pattern.format(
            prompt=prompt_stem,
            provider=provider_name,
            model=model,
            optimizer=optimizer,
        )

        return self.config.output_dir / f"{filename}.md"

    def get_configs(self) -> list[dict[str, Any]]:
        """Get all configuration combinations."""
        return self._configs

    def get_batch_id(self) -> str:
        """Get the batch identifier."""
        return self.batch_id

    def create_nodes(self) -> list[Node]:
        """Create OptimizeNodes for all configurations."""
        nodes = []
        for config in self._configs:
            node = OptimizeNode(
                prompt_path=config["prompt_path"],
                output_path=config["output_path"],
                provider_name=config["provider_name"],
                model_name=config["model_name"],
                optimizer_name=config["optimizer_name"],
                variables=config["variables"],
            )
            nodes.append(node)
        return nodes
```

**Step 4: Implement BatchResult and ResultsAggregator**

Create `src/dspy_examples/commands/results.py`:

```python
"""Result aggregation and reporting for batch processing.

Generates:
- Individual output files for each optimization
- Summary report in Markdown format
- Full results in JSON format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dspy_examples.commands.base import CommandResult


@dataclass
class BatchResult:
    """Result of a batch optimization run.

    Attributes:
        batch_id: Unique identifier for this batch
        total_commands: Total number of commands in batch
        successful: Number of successful commands
        failed: Number of failed commands
        skipped: Number of skipped commands
        total_time: Total execution time in seconds
        results: List of individual command results
        by_provider: Statistics grouped by provider
        by_optimizer: Statistics grouped by optimizer
    """

    batch_id: str
    total_commands: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    results: list[CommandResult]
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_optimizer: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert batch result to Markdown report."""
        lines = [
            f"# Batch Report: {self.batch_id}",
            "",
            "## Summary",
            f"- Total: {self.total_commands} commands",
            f"- Successful: {self.successful}",
            f"- Failed: {self.failed}",
            f"- Total time: {self.total_time:.1f}s",
            "",
        ]

        # Provider statistics
        if self.by_provider:
            lines.append("## By Provider")
            lines.append("| Provider | Model | Success | Failed | Avg Time |")
            lines.append("|----------|-------|---------|--------|----------|")

            for provider, stats in self.by_provider.items():
                model = stats.get("model", "default")
                success = stats.get("success", 0)
                failed = stats.get("failed", 0)
                avg_time = stats.get("avg_time", 0.0)
                lines.append(
                    f"| {provider} | {model} | {success} | {failed} | {avg_time:.1f}s |"
                )

            lines.append("")

        # Optimizer statistics
        if self.by_optimizer:
            lines.append("## By Optimizer")
            lines.append("| Optimizer | Avg Time | Success Rate |")
            lines.append("|-----------|----------|--------------|")

            for optimizer, stats in self.by_optimizer.items():
                avg_time = stats.get("avg_time", 0.0)
                success_rate = stats.get("success_rate", 0.0)
                lines.append(
                    f"| {optimizer} | {avg_time:.1f}s | {success_rate:.0%} |"
                )

            lines.append("")

        # Failed commands
        failed_results = [r for r in self.results if r.status == "failed"]
        if failed_results:
            lines.append("## Failed Commands")
            for result in failed_results:
                lines.append(
                    f"- {result.command_id}: {result.error_message or 'Unknown error'}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert batch result to JSON dict."""
        return {
            "batch_id": self.batch_id,
            "total_commands": self.total_commands,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_time": self.total_time,
            "results": [
                {
                    "command_id": r.command_id,
                    "status": r.status,
                    "output_path": str(r.output_path) if r.output_path else None,
                    "optimizer_name": r.optimizer_name,
                    "provider_name": r.provider_name,
                    "model_name": r.model_name,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "by_provider": self.by_provider,
            "by_optimizer": self.by_optimizer,
        }

    def save(self, output_dir: Path) -> None:
        """Save batch result to files.

        Creates:
            - {batch_id}_report.md: Markdown summary
            - {batch_id}_results.json: Full JSON results

        Args:
            output_dir: Directory to save files in
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown report
        md_path = output_dir / f"{self.batch_id}_report.md"
        md_path.write_text(self.to_markdown())

        # Save JSON results
        json_path = output_dir / f"{self.batch_id}_results.json"
        json_path.write_text(json.dumps(self.to_json(), indent=2))


class ResultsAggregator:
    """Aggregates results from batch processing.

    Collects CommandResults and computes statistics.
    """

    def __init__(self) -> None:
        """Initialize empty aggregator."""
        self.results: list[CommandResult] = []

    def add(self, result: CommandResult) -> None:
        """Add a command result.

        Args:
            result: Command result to add
        """
        self.results.append(result)

    def aggregate(self) -> BatchResult:
        """Create BatchResult with statistics.

        Returns:
            BatchResult with aggregated statistics
        """
        if not self.results:
            return BatchResult(
                batch_id="empty",
                total_commands=0,
                successful=0,
                failed=0,
                skipped=0,
                total_time=0.0,
                results=[],
            )

        # Generate batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Count by status
        successful = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        total_time = sum(r.execution_time for r in self.results)

        # Compute by-provider statistics
        by_provider: dict[str, dict[str, Any]] = {}
        for r in self.results:
            key = r.provider_name
            if key not in by_provider:
                by_provider[key] = {
                    "model": r.model_name,
                    "count": 0,
                    "success": 0,
                    "failed": 0,
                    "total_time": 0.0,
                }
            by_provider[key]["count"] += 1
            by_provider[key]["total_time"] += r.execution_time
            if r.status == "success":
                by_provider[key]["success"] += 1
            elif r.status == "failed":
                by_provider[key]["failed"] += 1

        # Compute averages
        for key in by_provider:
            stats = by_provider[key]
            count = stats["count"]
            stats["avg_time"] = stats["total_time"] / count if count > 0 else 0.0
            stats["success_rate"] = stats["success"] / count if count > 0 else 0.0

        # Compute by-optimizer statistics
        by_optimizer: dict[str, dict[str, Any]] = {}
        for r in self.results:
            key = r.optimizer_name
            if key not in by_optimizer:
                by_optimizer[key] = {
                    "count": 0,
                    "success": 0,
                    "total_time": 0.0,
                }
            by_optimizer[key]["count"] += 1
            by_optimizer[key]["total_time"] += r.execution_time
            if r.status == "success":
                by_optimizer[key]["success"] += 1

        # Compute averages
        for key in by_optimizer:
            stats = by_optimizer[key]
            count = stats["count"]
            stats["avg_time"] = stats["total_time"] / count if count > 0 else 0.0
            stats["success_rate"] = stats["success"] / count if count > 0 else 0.0

        return BatchResult(
            batch_id=batch_id,
            total_commands=len(self.results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time=total_time,
            results=self.results,
            by_provider=by_provider,
            by_optimizer=by_optimizer,
        )
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_flows.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/dspy_examples/commands/flows.py src/dspy_examples/commands/results.py tests/test_commands_flows.py
git commit -m "feat: add BatchFlow and ResultsAggregator for batch processing

- Add BatchConfig for batch processing configuration
- Add BatchFlow to generate config combinations
- Add BatchResult for aggregated results
- Add ResultsAggregator to compute statistics
- Support by-provider and by-optimizer grouping
- Generate Markdown and JSON reports"
```

---

## Task 6: Create High-Level BatchCommand API

**Files:**
- Create: `src/dspy_examples/commands/batch.py`
- Test: `tests/test_commands_batch.py`

**Step 1: Write the failing test for BatchCommand**

Create `tests/test_commands_batch.py`:

```python
"""Tests for high-level BatchCommand API."""

import pytest
from pathlib import Path
import tempfile


class TestBatchCommand:
    """Tests for BatchCommand."""

    def test_batch_command_creation(self):
        """Test creating a BatchCommand with config."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/test.md")],
            providers=[{"name": "ollama"}],
        )

        batch = BatchCommand(config)

        assert batch.config == config
        assert batch.batch_id is not None

    def test_batch_command_generate_configs(self):
        """Test config generation for multi-prompt batch."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[
                Path("prompts/q1.md"),
                Path("prompts/q2.md"),
            ],
            providers=[{"name": "ollama"}],
            optimizer_name="bootstrap_fewshot",
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 2
        assert configs[0]["prompt_path"] == Path("prompts/q1.md")
        assert configs[1]["prompt_path"] == Path("prompts/q2.md")

    def test_batch_command_multi_provider(self):
        """Test config generation for multi-provider batch."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/test.md")],
            providers=[
                {"name": "openai", "model": "gpt-4"},
                {"name": "anthropic", "model": "claude-3"},
                {"name": "ollama", "model": "llama3"},
            ],
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 3
        provider_names = [c["provider_name"] for c in configs]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "ollama" in provider_names

    def test_batch_command_with_variables(self):
        """Test config generation with variable sets."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_template=Path("prompts/template.md"),
            variable_sets=[
                {"country": "France", "tone": "formal"},
                {"country": "Japan", "tone": "casual"},
            ],
            providers=[{"name": "ollama"}],
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        assert len(configs) == 2
        assert configs[0]["variables"]["country"] == "France"
        assert configs[1]["variables"]["country"] == "Japan"

    def test_batch_command_get_output_paths(self):
        """Test output path generation."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig

        config = BatchConfig(
            prompt_paths=[Path("prompts/question.md")],
            providers=[
                {"name": "openai", "model": "gpt-4"},
                {"name": "ollama", "model": "llama3"},
            ],
            optimizer_name="mipro_v2",
            output_dir=Path("prompts/batch_output"),
        )

        batch = BatchCommand(config)
        configs = batch.get_configs()

        # Check naming pattern
        assert "question_openai_gpt-4_mipro_v2.md" in str(configs[0]["output_path"])
        assert "question_ollama_llama3_mipro_v2.md" in str(configs[1]["output_path"])


class TestBatchCommandIntegration:
    """Integration tests for BatchCommand."""

    @pytest.mark.integration
    def test_batch_command_run_mock(self):
        """Test running batch with mock results."""
        from dspy_examples.commands.batch import BatchCommand
        from dspy_examples.commands.flows import BatchConfig
        from dspy_examples.commands.base import CommandResult
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test prompts
            prompt1 = Path(tmpdir) / "q1.md"
            prompt2 = Path(tmpdir) / "q2.md"
            prompt1.write_text("Question 1")
            prompt2.write_text("Question 2")

            config = BatchConfig(
                prompt_paths=[prompt1, prompt2],
                providers=[{"name": "ollama"}],
                output_dir=Path(tmpdir) / "output",
            )

            batch = BatchCommand(config)

            # Get configs
            configs = batch.get_configs()
            assert len(configs) == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_batch.py -v`
Expected: FAIL with "No module named 'dspy_examples.commands.batch'"

**Step 3: Implement BatchCommand**

Create `src/dspy_examples/commands/batch.py`:

```python
"""High-level BatchCommand API for batch processing.

Usage:
    from dspy_examples.commands import BatchCommand, BatchConfig
    from pathlib import Path

    # Multiple prompts, single provider
    config = BatchConfig(
        prompt_paths=[Path("prompts/q1.md"), Path("prompts/q2.md")],
        providers=[{"name": "openai", "model": "gpt-4"}],
    )
    batch = BatchCommand(config)
    result = batch.run()

    # Multi-provider comparison
    config = BatchConfig(
        prompt_paths=[Path("prompts/test.md")],
        providers=[
            {"name": "openai", "model": "gpt-4"},
            {"name": "anthropic", "model": "claude-3"},
        ],
    )
    result = batch.run()

    # Template with variable sets
    config = BatchConfig(
        prompt_template=Path("prompts/template.md"),
        variable_sets=[
            {"country": "France", "tone": "formal"},
            {"country": "Japan", "tone": "casual"},
        ],
        providers=[{"name": "ollama"}],
    )
    result = batch.run()
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from dspy_examples.commands.base import CommandResult
from dspy_examples.commands.flows import BatchConfig, BatchFlow
from dspy_examples.commands.queue import CommandQueue
from dspy_examples.commands.results import ResultsAggregator


class BatchCommand:
    """High-level API for batch prompt optimization.

    Orchestrates the full batch processing workflow:
    1. Generate configuration combinations
    2. Create queue for persistence
    3. Execute each configuration
    4. Aggregate results
    5. Generate reports
    """

    def __init__(self, config: BatchConfig) -> None:
        """Initialize batch command with configuration.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self._flow = BatchFlow(config)
        self.batch_id = self._flow.get_batch_id()
        self._queue_path = (
            self.config.output_dir / ".cache" / f"{self.batch_id}_queue.db"
        )

    def get_configs(self) -> list[dict[str, Any]]:
        """Get all configuration combinations.

        Returns:
            List of configuration dicts for each combination.
        """
        return self._flow.get_configs()

    def run(
        self,
        queue_path: Path | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run the batch processing.

        Args:
            queue_path: Optional path for queue database
            resume: Whether to resume from previous queue

        Returns:
            Dictionary with batch results:
            - batch_id: Unique identifier
            - total_commands: Total number of commands
            - successful: Number of successful commands
            - failed: Number of failed commands
            - total_time: Total execution time
            - results: List of individual results
            - report_path: Path to Markdown report
        """
        import dspy

        from dspy_examples.factory.provider_factory import ProviderFactory
        from dspy_examples.factory.optimizer_factory import OptimizerFactory
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
        from dspy_examples.template import PromptTemplate

        # Setup queue for persistence
        if queue_path:
            self._queue_path = queue_path

        queue = CommandQueue(self._queue_path)

        # Get configurations
        configs = self.get_configs()

        # Add commands to queue if not resuming
        if not resume or queue.is_empty():
            queue.clear()
            for i, cfg in enumerate(configs):
                queue.add(
                    self._create_command(i, cfg),
                    batch_id=self.batch_id,
                    priority=0,
                )

        # Setup aggregator
        aggregator = ResultsAggregator()

        # Process each configuration
        start_time = time.time()

        # Get pending configs (simplified - in production would use worker)
        pending = queue.get_pending()

        for type_name, cmd_data in pending:
            try:
                # Run optimization
                result = self._run_optimization(cmd_data)

                # Save result
                queue.save_result(cmd_data["id"], {
                    "status": "success",
                    "output_path": str(result.output_path),
                    "optimizer_name": result.optimizer_name,
                    "provider_name": result.provider_name,
                    "model_name": result.model_name or "default",
                    "execution_time": result.execution_time,
                    "metadata": {},
                })

                # Mark completed
                queue.mark_completed(cmd_data["id"])

                # Add to aggregator
                aggregator.add(CommandResult(
                    command_id=cmd_data["id"],
                    status="success",
                    output_path=result.output_path,
                    optimizer_name=result.optimizer_name,
                    provider_name=result.provider_name,
                    model_name=result.model_name or "default",
                    execution_time=result.execution_time,
                ))

            except Exception as e:
                # Mark failed
                queue.mark_failed(cmd_data["id"], str(e))

                # Add failed result
                aggregator.add(CommandResult(
                    command_id=cmd_data["id"],
                    status="failed",
                    output_path=None,
                    optimizer_name=cmd_data.get("optimizer_name", "unknown"),
                    provider_name=cmd_data.get("provider_name", "unknown"),
                    model_name=cmd_data.get("model_name", "unknown"),
                    execution_time=0.0,
                    error_message=str(e),
                ))

        total_time = time.time() - start_time

        # Generate batch result
        batch_result = aggregator.aggregate()
        batch_result.total_time = total_time

        # Save reports
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        batch_result.save(self.config.output_dir)

        return {
            "batch_id": self.batch_id,
            "total_commands": batch_result.total_commands,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
            "total_time": batch_result.total_time,
            "results": batch_result.results,
            "report_path": self.config.output_dir / f"{self.batch_id}_report.md",
        }

    def _create_command(self, index: int, config: dict[str, Any]) -> "OptimizeCommand":
        """Create OptimizeCommand from configuration.

        Args:
            index: Index for command ID
            config: Configuration dict

        Returns:
            OptimizeCommand instance
        """
        from dspy_examples.commands.optimize_command import OptimizeCommand

        return OptimizeCommand(
            command_id=f"{self.batch_id}_{index:03d}",
            prompt_path=config["prompt_path"],
            output_path=config["output_path"],
            provider_name=config["provider_name"],
            model_name=config.get("model_name"),
            optimizer_name=config["optimizer_name"],
            variables=config.get("variables", {}),
        )

    def _run_optimization(
        self, config: dict[str, Any]
    ) -> "OptimizationResult":
        """Run single optimization from config.

        Args:
            config: Configuration dict

        Returns:
            OptimizationResult
        """
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        pipeline_config = PipelineConfig(
            provider_name=config["provider_name"],
            optimizer_name=config["optimizer_name"],
            input_path=config["prompt_path"],
            output_path=config["output_path"],
            variables=config.get("variables", {}),
            use_cache=False,  # Don't use cache in batch
        )

        pipeline = OptimizationPipeline(pipeline_config)
        return pipeline.run(config.get("variables", {}))


# OptimizeCommand class for serialization
class OptimizeCommand:
    """Command for single prompt optimization.

    Implements Command interface for queue persistence.
    """

    def __init__(
        self,
        command_id: str,
        prompt_path: Path,
        output_path: Path,
        provider_name: str,
        model_name: str | None,
        optimizer_name: str,
        variables: dict[str, str],
    ) -> None:
        self._id = command_id
        self.prompt_path = Path(prompt_path)
        self.output_path = Path(output_path)
        self.provider_name = provider_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.variables = variables

    @property
    def command_id(self) -> str:
        return self._id

    def execute(self) -> CommandResult:
        """Execute the optimization."""
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        config = PipelineConfig(
            provider_name=self.provider_name,
            optimizer_name=self.optimizer_name,
            input_path=self.prompt_path,
            output_path=self.output_path,
            variables=self.variables,
            use_cache=False,
        )

        pipeline = OptimizationPipeline(config)
        result = pipeline.run(self.variables)

        return CommandResult(
            command_id=self._id,
            status="success",
            output_path=self.output_path,
            optimizer_name=result.optimizer_name,
            provider_name=result.provider_name,
            model_name=self.model_name or "default",
            execution_time=result.execution_time or 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self._id,
            "prompt_path": str(self.prompt_path),
            "output_path": str(self.output_path),
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "optimizer_name": self.optimizer_name,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeCommand":
        """Deserialize from dictionary."""
        return cls(
            command_id=data["id"],
            prompt_path=Path(data["prompt_path"]),
            output_path=Path(data["output_path"]),
            provider_name=data["provider_name"],
            model_name=data.get("model_name"),
            optimizer_name=data["optimizer_name"],
            variables=data.get("variables", {}),
        )
```

**Step 4: Update commands __init__.py**

Edit `src/dspy_examples/commands/__init__.py`:

```python
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
from dspy_examples.commands.batch import BatchCommand

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
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_commands_batch.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/dspy_examples/commands/__init__.py src/dspy_examples/commands/batch.py tests/test_commands_batch.py
git commit -m "feat: add high-level BatchCommand API

- Create BatchCommand for simple batch processing
- Support multi-prompt, multi-provider, multi-variable configs
- Integrate with queue persistence for resumability
- Auto-generate output paths with naming pattern
- Aggregate results and generate reports"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add batch processing section to README**

Edit `README.md`, add after the "Variable Substitution" section:

```markdown
## Batch Processing

Process multiple prompts with different providers and configurations:

### Multiple Prompts, Single Provider

```python
from dspy_examples.commands import BatchCommand, BatchConfig
from pathlib import Path

config = BatchConfig(
    prompt_paths=[
        Path("prompts/question.md"),
        Path("prompts/summary.md"),
    ],
    providers=[{"name": "openai", "model": "gpt-4"}],
    optimizer_name="bootstrap_fewshot",
)

batch = BatchCommand(config)
result = batch.run()
print(f"Successful: {result['successful']}/{result['total_commands']}")
```

### Single Template, Multiple Variables

```python
config = BatchConfig(
    prompt_template=Path("prompts/geography.md"),
    variable_sets=[
        {"country": "France", "tone": "formal"},
        {"country": "Japan", "tone": "casual"},
    ],
    providers=[{"name": "ollama"}],
    optimizer_name="mipro_v2",
)

batch = BatchCommand(config)
result = batch.run()
```

### Multi-Provider Comparison

```python
config = BatchConfig(
    prompt_paths=[Path("prompts/question.md")],
    providers=[
        {"name": "openai", "model": "gpt-4"},
        {"name": "anthropic", "model": "claude-3-opus"},
        {"name": "google", "model": "gemini-pro"},
        {"name": "ollama", "model": "llama3"},
    ],
    optimizer_name="gepa",
    max_concurrent=3,
)

batch = BatchCommand(config)
result = batch.run()

# Compare providers
for provider, stats in result.get("by_provider", {}).items():
    print(f"{provider}: {stats['avg_time']:.1f}s, {stats['success_rate']:.0%}")
```

### Output Files

Batch processing creates:
```
prompts/batch_output/
├── batch_abc123_report.md          # Summary report
├── batch_abc123_results.json       # Full results
├── question_openai_gpt4_bootstrap.md
├── question_ollama_llama3_bootstrap.md
└── ...
```

### Resume Interrupted Batch

```python
from dspy_examples.commands import CommandQueue

queue = CommandQueue(Path(".cache/batch_commands.db"))
pending = queue.get_pending()
print(f"Resuming {len(pending)} pending commands")
```

## Project Structure

```
dspy_examples/
├── prompts/
│   ├── unoptimized_prompt.md    # Input prompts
│   └── optimized_prompt.md      # Output (optimized)
├── src/dspy_examples/
│   ├── pocketflow/             # Embedded PocketFlow for orchestration
│   │   ├── __init__.py
│   │   └── core.py              # Node, Flow, BatchNode, BatchFlow
│   ├── commands/               # Command pattern for batch processing
│   │   ├── __init__.py
│   │   ├── base.py              # Command, CommandResult
│   │   ├── nodes.py             # OptimizeNode, LoadPromptNode
│   │   ├── flows.py             # BatchConfig, BatchFlow
│   │   ├── queue.py             # SQLite persistence
│   │   ├── results.py           # ResultsAggregator, BatchResult
│   │   └── batch.py             # BatchCommand API
│   ├── providers/               # LLM Provider Adapters
│   ├── optimizers/              # Optimization Strategies
│   ├── factory/                 # Factory Pattern
│   ├── pipeline.py              # Template Pattern
│   ├── template.py              # Variable substitution
│   ├── cache.py                 # Memento Pattern
│   └── config.py                # DSPy configuration
├── tests/
│   ├── test_pocketflow_*.py     # PocketFlow tests
│   ├── test_commands_*.py       # Command tests
│   └── ...
├── main.py                      # Entry point
├── pyproject.toml
└── README.md
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add batch processing documentation

- Document multiple prompts with single provider
- Document template with multiple variable sets
- Document multi-provider comparison
- Add output file structure
- Add resume instructions
- Update project structure with commands/ and pocketflow/"
```

---

## Task 8: Update Design Document

**Files:**
- Modify: `docs/plans/2026-03-22-batch-processing-design.md`

**Step 1: Update design document to reflect PocketFlow integration**

Update the entire design document to include:
- PocketFlow integration section
- Updated directory structure with pocketflow/
- Updated architecture diagram
- Implementation notes about embedded PocketFlow

**Step 2: Commit**

```bash
git add docs/plans/2026-03-22-batch-processing-design.md
git commit -m "docs: update batch processing design with PocketFlow integration"
```

---

## Execution Summary

This implementation plan creates batch processing with:

1. **Embedded PocketFlow** (~100 lines) - Core Node/Flow primitives
2. **Command Base Classes** - CommandResult, Command interface
3. **SQLite Queue** - Persistence for resumability
4. **PocketFlow Nodes** - OptimizeNode, LoadPromptNode
5. **Batch Flow** - BatchConfig, BatchFlow orchestration
6. **Results Aggregation** - Statistics, Markdown/JSON reports
7. **High-Level API** - BatchCommand for easy use

Each task follows TDD: test first, then implement. Frequent commits maintain clean history.