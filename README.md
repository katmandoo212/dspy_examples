# DSPy Examples

A Python project demonstrating DSPy prompt optimization techniques using Gang of Four design patterns for extensibility.

## Features

- **Multiple Optimizers**: BootstrapFewShot, BootstrapRandom, MIPROv2, GEPA, BetterTogether, COPRO, BootstrapFinetune, SIMBA
- **Multiple LLM Providers**: Ollama, OpenAI, Anthropic, Google, OpenRouter
- **Variable Substitution**: Configurable placeholders with preserve/substitute modes
- **Factory Pattern**: Easy creation of providers and optimizers
- **Pipeline Pattern**: Standard optimization workflow with caching
- **Extensible Architecture**: Easy to add new optimizers and providers

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Optimization Pipeline                         │
│       Pipeline (Template) ──── Cache (Memento)                  │
│              │                                                   │
│       ProviderFactory ──── OptimizerFactory                       │
│              │                    │                              │
│       ┌──────┴──────┐      ┌──────┴──────┐                      │
│       │  Providers   │      │  Optimizers │                      │
│       │  (Adapter)   │      │  (Strategy) │                      │
│       └─────────────┘      └─────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

| Pattern | Component | Purpose |
|---------|-----------|---------|
| **Adapter** | Providers | Unified interface for multiple LLM backends |
| **Strategy** | Optimizers | Swap optimization algorithms |
| **Factory** | Factories | Create providers and optimizers by name |
| **Template** | Pipeline | Standard optimization workflow |
| **Memento** | Cache | Save/restore optimization results |

## Setup

1. Install [uv](https://docs.astral.sh/uv/) or use pip
2. Install dependencies:
   ```bash
   uv sync
   # or
   pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and configure:
   ```
   # Choose your LLM provider
   LLM_PROVIDER=ollama  # or openai, anthropic, google, openrouter

   # Ollama Configuration
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gpt-oss:120b-cloud

   # OpenAI Configuration (if using openai)
   OPENAI_API_KEY=your-key-here
   OPENAI_MODEL=gpt-4

   # Anthropic Configuration (if using anthropic)
   ANTHROPIC_API_KEY=your-key-here
   ANTHROPIC_MODEL=claude-3-opus-20240229

   # Google Configuration (if using google)
   GOOGLE_API_KEY=your-key-here
   GOOGLE_MODEL=gemini-pro

   # OpenRouter Configuration (if using openrouter)
   OPENROUTER_API_KEY=your-key-here
   OPENROUTER_MODEL=anthropic/claude-3-opus
   ```

4. If using Ollama, ensure it's running:
   ```bash
   ollama run gpt-oss:120b-cloud
   ```

## Usage

### Basic Usage

```python
from dspy_examples.pipeline import OptimizationPipeline

# Auto-select optimizer based on prompt length
pipeline = OptimizationPipeline()
result = pipeline.run()
print(f"Optimized with: {result.optimizer_name}")
```

### Explicit Configuration

```python
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
from pathlib import Path

pipeline = OptimizationPipeline(PipelineConfig(
    provider_name="openai",
    optimizer_name="bootstrap_random",
    input_path=Path("prompts/unoptimized_prompt.md"),
    output_path=Path("prompts/optimized_prompt.md"),
))
result = pipeline.run()
```

### Using Factory Directly

```python
from dspy_examples.factory.provider_factory import ProviderFactory
from dspy_examples.factory.optimizer_factory import OptimizerFactory

# Create a provider
provider = ProviderFactory.create("ollama")
lm = provider.create_lm()

# Create an optimizer
optimizer = OptimizerFactory.create("bootstrap_random")

# List available options
print(ProviderFactory.list_providers())
print(OptimizerFactory.list_optimizers())
```

### Variable Substitution

Prompts can contain variables using configurable delimiters (default `[[ ]]`):

```markdown
---
delimiter: "[[ ]]"
variables:
  country:
    mode: preserve
    default: USA
  tone:
    mode: substitute
    default: professional
---

You are a geography expert. Answer questions about [[country]] in a [[tone]] manner.
```

**Variable Modes:**
- `preserve`: Placeholder retained in optimized output
- `substitute`: Permanently replaced with value

**Usage:**

```python
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

# Variables from config
config = PipelineConfig(
    input_path=Path("prompts/template.md"),
    variables={"country": "Japan"}
)

# Or variables at runtime
pipeline = OptimizationPipeline(config)
result = pipeline.run(variables={"country": "France", "tone": "casual"})
```

**Delimiter Configuration:**

Project-wide default in `.env`:
```
VARIABLE_DELIMITER_START=[[
VARIABLE_DELIMITER_END=]]
```

Per-prompt override in frontmatter:
```yaml
---
delimiter: "${ }"
---

Discuss ${topic} in detail.
```

**PromptTemplate API:**

```python
from dspy_examples.template import PromptTemplate

# Load and parse
template = PromptTemplate.from_file("prompt.md")

# Validate
errors = template.validate({"country": "France"})
if errors:
    print(f"Validation errors: {errors}")

# Substitute
prompt = template.substitute({"country": "France", "tone": "formal"})

# Check preserved variables
preserved = template.get_preserved_variables()  # ['country']
```

### Batch Processing

Process multiple prompts with different providers and configurations:

#### Multiple Prompts, Single Provider

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

#### Single Template, Multiple Variables

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

#### Multi-Provider Comparison

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

#### Output Files

Batch processing creates:
```
prompts/batch_output/
├── batch_abc123_report.md          # Summary report
├── batch_abc123_results.json       # Full results
├── question_openai_gpt4_bootstrap.md
├── question_ollama_llama3_bootstrap.md
└── ...
```

#### Resume Interrupted Batch

```python
from dspy_examples.commands import CommandQueue
from pathlib import Path

queue = CommandQueue(Path(".cache/batch_commands.db"))
pending = queue.get_pending()
print(f"Resuming {len(pending)} pending commands")
```

### Available Optimizers

| Optimizer | Name | Description |
|----------|------|-------------|
| BootstrapFewShot | `bootstrap_fewshot` | Adds few-shot examples using DSPy's teleprompter |
| BootstrapRandom | `bootstrap_random` | Uses random search to find optimal demonstration combinations |
| MIPROv2 | `mipro_v2` | Uses Bayesian optimization for instructions and few-shot examples |
| GEPA | `gepa` | Reflective prompt evolution with Pareto-based candidate selection |
| BetterTogether | `better_together` | Combines prompt optimization with fine-tuning for compound improvements |
| COPRO | `copro` | Coordinate ascent for instruction optimization (hill-climbing) |
| BootstrapFinetune | `bootstrap_finetune` | Distills prompts into weight updates for fine-tuning |
| SIMBA | `simba` | Stochastic mini-batch ascent with self-reflective improvement |

### Available Providers

| Provider | Name | Environment Variables |
|----------|------|----------------------|
| Ollama | `ollama` | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| Google | `google` | `GOOGLE_API_KEY`, `GOOGLE_MODEL` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |

## Project Structure

```
dspy_examples/
├── prompts/
│   ├── unoptimized_prompt.md    # Input prompts
│   └── optimized_prompt.md      # Output (optimized)
├── src/dspy_examples/
│   ├── pocketflow/              # Embedded PocketFlow for orchestration
│   │   ├── __init__.py
│   │   └── core.py              # Node, Flow, BatchNode, BatchFlow
│   ├── commands/                # Command pattern for batch processing
│   │   ├── __init__.py
│   │   ├── base.py              # Command, CommandResult
│   │   ├── nodes.py             # OptimizeNode, LoadPromptNode
│   │   ├── flows.py             # BatchConfig, BatchFlow
│   │   ├── queue.py             # SQLite persistence
│   │   ├── results.py           # ResultsAggregator, BatchResult
│   │   └── batch.py             # BatchCommand API
│   ├── providers/               # LLM Provider Adapters
│   │   ├── base.py              # LMProvider interface
│   │   ├── ollama.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   └── openrouter.py
│   ├── optimizers/              # Optimization Strategies
│   │   ├── base.py              # PromptOptimizer interface
│   │   ├── bootstrap_fewshot.py
│   │   ├── bootstrap_random.py
│   │   ├── mipro_v2.py
│   │   ├── gepa.py
│   │   ├── better_together.py
│   │   ├── copro.py
│   │   ├── bootstrap_finetune.py
│   │   └── simba.py
│   ├── factory/                 # Factory Pattern
│   │   ├── provider_factory.py
│   │   └── optimizer_factory.py
│   ├── pipeline.py              # Template Pattern
│   ├── template.py               # Variable substitution
│   ├── cache.py                 # Memento Pattern
│   ├── config.py                # DSPy configuration
│   ├── settings.py              # Pydantic Settings
│   ├── prompts.py               # Prompt I/O utilities
│   └── bootstrap_fewshot.py     # Legacy entry point
├── tests/
│   ├── test_pocketflow_*.py     # PocketFlow tests
│   ├── test_commands_*.py       # Command tests
│   ├── test_providers_*.py      # Provider tests
│   ├── test_optimizers_*.py     # Optimizer tests
│   ├── test_factory_*.py        # Factory tests
│   └── test_pipeline.py         # Pipeline tests
├── main.py                      # Entry point
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
# Run all tests
PYTHONPATH=src uv run pytest tests/ -v

# Run unit tests only (skip integration tests)
PYTHONPATH=src uv run pytest tests/ -v -m "not integration"

# Run with coverage
PYTHONPATH=src uv run pytest tests/ --cov=src/dspy_examples
```

## Extending the Project

### Adding a New Provider

```python
# src/dspy_examples/providers/my_provider.py
from dspy_examples.providers.base import LMProvider, LMConfig

class MyProvider(LMProvider):
    @classmethod
    def from_env(cls) -> "MyProvider":
        # Load config from environment
        ...

    def create_lm(self) -> dspy.LM:
        # Create and return DSPy LM
        ...

    def validate_config(self) -> bool:
        # Validate configuration
        ...

    @classmethod
    def provider_name(cls) -> str:
        return "my_provider"

# Register in factory
from dspy_examples.factory.provider_factory import ProviderFactory
ProviderFactory.register("my_provider", MyProvider)
```

### Adding a New Optimizer

```python
# src/dspy_examples/optimizers/my_optimizer.py
from dspy_examples.optimizers.base import PromptOptimizer, OptimizationResult, OptimizerConfig

class MyOptimizer(PromptOptimizer):
    def __init__(self, config: OptimizerConfig | None = None):
        self._config = config or OptimizerConfig(name="my_optimizer")

    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        # Implement optimization
        ...

    def get_name(self) -> str:
        return "my_optimizer"

    def get_description(self) -> str:
        return "My custom optimizer"

    def get_config(self) -> OptimizerConfig:
        return self._config

# Register in factory
from dspy_examples.factory.optimizer_factory import OptimizerFactory
OptimizerFactory.register("my_optimizer", MyOptimizer)
```

## Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Adapter** | `providers/` | Multiple LLM backends with unified interface |
| **Strategy** | `optimizers/` | Swap optimization algorithms |
| **Factory** | `factory/` | Create providers & optimizers by name |
| **Template** | `pipeline.py` | Standard optimization workflow |
| **Memento** | `cache.py` | Save/restore results |
| **Command** | `commands/` | Batch processing with queue persistence |
| **Observer** | `observers/` | (Future) Progress tracking |
| **Builder** | `builders/` | (Future) Complex configurations |

## License

MIT