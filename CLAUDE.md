# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

DSPy prompt optimization examples - a collection of Python scripts demonstrating DSPy techniques for optimizing prompts using BootstrapFewShot and other methods.

## Environment

- **Python**: 3.13+ (specified in `.python-version`)
- **Package Manager**: uv
- **Virtual Environment**: Shared at `G:\home\blt\github\.venv`

## Development Commands

### Run Tests

```bash
# Run all tests
PYTHONPATH=src uv run pytest tests/ -v

# Run unit tests only (skip integration tests)
PYTHONPATH=src uv run pytest tests/ -v -m "not integration"

# Run with coverage
PYTHONPATH=src uv run pytest tests/ --cov=src/dspy_examples
```

### Install Dependencies

```bash
uv sync
```

### Run the Optimization

```bash
# Recommended: Use uv run (handles src/ layout automatically)
uv run python main.py

# Alternative: With package installed
uv sync && python main.py
```

## Project Structure

```
dspy_examples/
├── prompts/
│   ├── unoptimized_prompt.md    # Input prompt file
│   └── optimized_prompt.md      # Output (auto-generated)
├── src/dspy_examples/
│   ├── __init__.py              # Package init
│   ├── config.py                # DSPy/Ollama configuration
│   ├── prompts.py               # Prompt I/O utilities
│   └── bootstrap_fewshot.py     # BootstrapFewShot implementation
├── tests/
│   ├── test_config.py           # Config module tests
│   ├── test_prompts.py           # Prompts utility tests
│   ├── test_bootstrap_fewshot.py # BootstrapFewShot tests
│   └── test_main.py             # Main entry point tests
├── main.py                      # Entry point
├── pyproject.toml               # Project configuration
├── .env.example                 # Environment template
└── README.md                    # Documentation
```

## DSPy-Specific Notes

### Testing with DSPy

DSPy requires actual language model instances for most operations. Key testing patterns:

1. **Unit Tests**: Test module structure, signatures, and utilities without an LM
2. **Integration Tests**: Mark with `@pytest.mark.integration` and skip in CI
3. **Mock Limitations**: `MagicMock` doesn't work because DSPy validates `isinstance(lm, dspy.BaseLM)`

### Test Isolation

When testing modules that use `dspy.configure()`:
- Use `@patch("dspy.configure")` to mock global configuration
- Use `@patch.dict(os.environ, {}, clear=True)` to isolate environment variables

### BootstrapFewShot

**Critical: Training examples must specify inputs with `.with_inputs()`**

```python
# WRONG - Will fail with "Inputs have not been set for this example"
trainset=[
    dspy.Example(question="What is 2+2?", answer="4")
]

# CORRECT - Specify which fields are inputs
trainset=[
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")
]
```

**Critical: Metric function requires 3 arguments**

The metric function signature must be `metric(example, prediction, trace=None)`:

```python
# WRONG - Only 2 arguments, causes "takes 2 positional arguments but 3 were given"
metric=lambda x, y: True

# CORRECT - 3 arguments with optional trace
def metric(example, prediction, trace=None):
    """Evaluate if the prediction is acceptable."""
    return True  # Accept all for demo purposes

# Production metric - evaluate quality
def quality_metric(example, prediction, trace=None):
    # Check if prediction matches expected output
    return prediction.answer == example.answer
```

**Complete BootstrapFewShot example:**

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, prediction, trace=None):
    # Evaluate prediction quality
    return True

optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
)

optimized = optimizer.compile(
    my_module,
    trainset=[
        dspy.Example(
            unoptimized_prompt="Write a greeting.",
            optimized_prompt="Write a greeting.\n\nExample: ..."
        ).with_inputs("unoptimized_prompt")
    ],
)
```

## Adding New Optimization Techniques

Each technique should be a self-contained module:

1. Create `src/dspy_examples/new_technique.py` with:
   - `dspy.Signature` class defining inputs/outputs
   - `dspy.Module` class with `forward()` method
   - `optimize_prompt()` or similar main function
   - `main()` function for CLI usage

2. Create `tests/test_new_technique.py` with:
   - Tests for signature fields
   - Tests for module instantiation
   - Integration tests marked with `@pytest.mark.integration`

3. Update `main.py` to call the new technique

4. Add documentation to `README.md`

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud
```

### Ollama Setup

Ensure Ollama is running with the configured model:
```bash
ollama run gpt-oss:120b-cloud
```

## Code Style

- Use type hints (`dict[str, Any]`, `str | Path`, etc.)
- Follow PEP 8 naming conventions
- Add docstrings with Args, Returns, and Raises sections
- Use `pathlib.Path` for file operations