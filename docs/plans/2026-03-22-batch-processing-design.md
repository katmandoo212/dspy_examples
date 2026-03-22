# Batch Processing with Command Pattern - Design Document

**Date**: 2026-03-22
**Status**: Implemented

## Summary

Add batch processing using the Command Design Pattern with SQLite-backed queue persistence and embedded PocketFlow for execution orchestration. Supports multiple prompts, multiple variable sets, and multi-provider execution with individual output files plus aggregated summary reports.

## Requirements

### Functional Requirements

1. **Multiple prompts, single config** - Process a folder of prompt files with the same optimizer/provider settings
2. **Single prompt, multiple variables** - Process one template prompt with different variable values
3. **Multi-provider support** - Run same prompt on multiple providers (OpenAI, Anthropic, Google, Ollama, OpenRouter) with different models
4. **Queue-based execution** - Commands added to queue, executed by workers
5. **Resumable processing** - SQLite persistence for recovery after crashes
6. **Result aggregation** - Individual output files plus summary report (Markdown and JSON)

### Non-Functional Requirements

- Thread-safe queue operations
- Configurable concurrency
- Retry support for failed commands
- Per-command timing and metadata
- Provider/model comparison in summary

## Architecture

### Directory Structure

```
commands/
├── __init__.py
├── base.py              # Command abstract base class
├── optimize_command.py  # Concrete: optimize a prompt
├── batch_command.py     # Concrete: run a batch of commands
├── queue.py             # SQLite-backed priority queue
├── worker.py            # Worker that processes queue
└── results.py           # Result aggregation and reporting
```

### Core Components

1. **Command** - Encapsulates a single optimization request
2. **CommandQueue** - SQLite-backed queue with persistence
3. **Worker** - Pulls commands from queue, executes, reports results
4. **BatchCommand** - Creates and manages batches of commands
5. **ResultsAggregator** - Collects results, generates reports

## Data Structures

### CommandResult

```python
@dataclass
class CommandResult:
    command_id: str
    status: Literal["success", "failed", "skipped"]
    output_path: Path | None
    optimizer_name: str
    provider_name: str
    model_name: str
    execution_time: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Command (Base)

```python
class Command(ABC):
    @property
    @abstractmethod
    def command_id(self) -> str: ...

    @abstractmethod
    def execute(self) -> CommandResult: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Command": ...
```

### OptimizeCommand

```python
@dataclass
class OptimizeCommand(Command):
    prompt_path: Path
    output_path: Path
    provider_name: str
    model_name: str | None
    optimizer_name: str
    variables: dict[str, str] | None
    config: OptimizerConfig | None
```

### BatchConfig

```python
@dataclass
class BatchConfig:
    # Prompt sources
    prompt_paths: list[Path] | None = None
    prompt_template: Path | None = None
    variable_sets: list[dict[str, str]] | None = None

    # Provider configurations
    providers: list[dict[str, str]] = field(default_factory=lambda: [
        {"name": "ollama"}
    ])

    # Optimizer settings
    optimizer_name: str = "bootstrap_fewshot"
    optimizer_config: OptimizerConfig | None = None

    # Output settings
    output_dir: Path = Path("prompts/batch_output")
    naming_pattern: str = "{prompt}_{provider}_{model}_{optimizer}"

    # Execution settings
    max_concurrent: int = 1
    retry_count: int = 2
```

### BatchResult

```python
@dataclass
class BatchResult:
    batch_id: str
    total_commands: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    results: list[CommandResult]
    by_provider: dict[str, dict[str, Any]]
    by_optimizer: dict[str, dict[str, Any]]

    def to_markdown(self) -> str: ...
    def to_json(self) -> dict: ...
    def save(self, path: Path): ...
```

## Database Schema

```sql
CREATE TABLE commands (
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

CREATE TABLE results (
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

CREATE INDEX idx_status ON commands(status);
CREATE INDEX idx_batch ON commands(batch_id);
```

## Output File Structure

```
prompts/batch_output/
├── batch_abc123_report.md          # Summary report (Markdown)
├── batch_abc123_results.json       # Full results (JSON)
├── question_openai_gpt4_bootstrap.md
├── question_ollama_llama3_bootstrap.md
├── question_anthropic_claude_mipro.md
└── ...
```

## Summary Report Format

```markdown
# Batch Report: abc123

## Summary
- Total: 12 commands
- Successful: 11
- Failed: 1
- Total time: 342.5s

## By Provider
| Provider | Model | Success | Failed | Avg Time |
|----------|-------|---------|--------|----------|
| openai | gpt-4 | 4/4 | 0 | 28.3s |
| ollama | llama3 | 3/4 | 1 | 35.2s |
| anthropic | claude-3 | 4/4 | 0 | 22.1s |

## By Optimizer
| Optimizer | Avg Time | Success Rate |
|-----------|----------|--------------|
| bootstrap_fewshot | 25.1s | 100% |
| mipro_v2 | 38.2s | 91% |

## Failed Commands
- prompt1_ollama_llama3_bootstrap: Connection timeout
```

## Usage Examples

### Multiple Prompts, Single Provider

```python
from dspy_examples.commands import BatchCommand, BatchConfig

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
result.save(Path("prompts/batch_output"))
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

for provider, stats in result.by_provider.items():
    print(f"{provider}: {stats['avg_time']:.1f}s, {stats['success_rate']:.0%}")
```

### Resume Interrupted Batch

```python
from dspy_examples.commands import CommandQueue, Worker

queue = CommandQueue(Path(".cache/batch_commands.db"))
pending = queue.get_pending()
print(f"Resuming {len(pending)} pending commands")

worker = Worker(queue)
worker.process_all()
```

## Implementation Order

1. Create `commands/base.py` with Command and CommandResult
2. Create `commands/queue.py` with SQLite-backed CommandQueue
3. Create `commands/optimize_command.py` with OptimizeCommand
4. Create `commands/worker.py` with Worker
5. Create `commands/results.py` with ResultsAggregator
6. Create `commands/batch_command.py` with BatchCommand
7. Add unit tests for each component
8. Add integration tests for batch execution
9. Update README with batch processing documentation

## Test Strategy

### Unit Tests

- `test_commands_base.py` - Command/CommandResult classes
- `test_commands_queue.py` - SQLite queue operations
- `test_commands_optimize.py` - OptimizeCommand execution
- `test_commands_worker.py` - Worker processing logic
- `test_commands_results.py` - Result aggregation

### Integration Tests

- `test_commands_batch.py` - End-to-end batch execution
- `test_commands_resume.py` - Resume from interrupted batch

## File Organization

```
src/dspy_examples/commands/
├── __init__.py           # Public exports
├── base.py               # Command, CommandResult
├── optimize_command.py   # OptimizeCommand
├── batch_command.py      # BatchConfig, BatchCommand
├── queue.py              # CommandQueue (SQLite)
├── worker.py             # Worker
└── results.py            # ResultsAggregator, BatchResult

tests/
├── test_commands_base.py
├── test_commands_queue.py
├── test_commands_optimize.py
├── test_commands_worker.py
├── test_commands_results.py
├── test_commands_batch.py
└── test_commands_resume.py
```