# DSPy Examples

A collection of Python scripts demonstrating DSPy prompt optimization techniques.

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
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gpt-oss:120b-cloud
   ```

4. Ensure Ollama is running with your model:
   ```bash
   ollama run gpt-oss:120b-cloud
   ```

## Usage

### BootstrapFewShot Optimization

After running `uv sync` (which installs the package), use one of these methods:

```bash
# Method 1: Using uv run (recommended)
uv run python main.py

# Method 2: Using the installed package
uv run python -m dspy_examples.bootstrap_fewshot

# Method 3: With pip-installed environment
python main.py
# or
python -m dspy_examples.bootstrap_fewshot
```

> **Note**: This project uses a `src/` layout. The package must be installed (via `uv sync` or `pip install -e .`) for Python to find it. If you see `ModuleNotFoundError: No module named 'dspy_examples'`, run `uv sync` first.

This reads from `prompts/unoptimized_prompt.md` and writes the optimized version to `prompts/optimized_prompt.md` (or `optimized_prompt_XX.md` if file exists).

## Project Structure

```
dspy_examples/
├── prompts/
│   ├── unoptimized_prompt.md    # Input prompts
│   └── optimized_prompt.md      # Output (optimized)
├── src/dspy_examples/
│   ├── __init__.py
│   ├── config.py                # DSPy configuration
│   ├── prompts.py               # Prompt I/O utilities
│   └── bootstrap_fewshot.py     # BootstrapFewShot optimization
├── tests/
│   ├── test_config.py
│   ├── test_prompts.py
│   ├── test_bootstrap_fewshot.py
│   └── test_main.py
├── main.py                      # Entry point
├── pyproject.toml
└── README.md
```

## Adding New Optimization Techniques

Each technique should be a self-contained script in `src/dspy_examples/`:

1. Create module: `src/dspy_examples/new_technique.py`
2. Create tests: `tests/test_new_technique.py`
3. Add entry point to `main.py` or run directly

## Running Tests

```bash
# Run all tests (using uv run handles the src/ layout)
uv run pytest tests/ -v

# Alternative: Set PYTHONPATH manually
PYTHONPATH=src pytest tests/ -v

# Run unit tests only (skip integration tests)
uv run pytest tests/ -v -m "not integration"

# Run with coverage
uv run pytest tests/ --cov=src/dspy_examples
```

## License

MIT