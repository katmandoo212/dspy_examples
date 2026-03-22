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

    def get_pending(self, limit: int | None = None) -> list[tuple[str, dict[str, Any]]]:
        """Get pending commands from queue.

        Args:
            limit: Maximum number of commands to return

        Returns:
            List of tuples (command_id, data_dict) for pending commands,
            ordered by priority desc, created_at asc.
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT id, data FROM commands
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """
            if limit:
                query += f" LIMIT {limit}"

            rows = conn.execute(query).fetchall()

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
        """Get all completed commands.

        Returns command data from commands table. For result data,
        use save_result() before marking completed.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, batch_id, type, data, completed_at
                FROM commands
                WHERE status = 'completed'
                ORDER BY completed_at DESC
                """
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "batch_id": row[1],
                    "type": row[2],
                    "data": json.loads(row[3]) if row[3] else {},
                    "completed_at": row[4],
                }
                for row in rows
            ]

    def get_failed(self) -> list[dict[str, Any]]:
        """Get all failed commands with error messages."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.batch_id, r.error_message
                FROM commands c
                JOIN results r ON c.id = r.command_id
                WHERE c.status = 'failed'
                ORDER BY c.completed_at DESC
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