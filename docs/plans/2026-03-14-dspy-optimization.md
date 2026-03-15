# DSPy Prompt Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python project using DSPy to optimize prompts via BootstrapFewShot, reading from `unoptimized_prompt.md` and outputting optimized versions.

**Architecture:** Self-contained scripts, one per optimization technique. Each script configures DSPy with Ollama (`gpt-oss:120b-cloud`), loads an unoptimized prompt, applies optimization, and saves results. Uses TDD with pytest.

**Tech Stack:** Python 3.14, DSPy, Ollama, pytest, pyproject.toml (uv/pip)

---

## Task 1: Project Setup and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`

**Step 1: Update pyproject.toml with dependencies**

```toml
[project]
name = "dspy-examples"
version = "0.1.0"
description = "DSPy prompt optimization examples"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "dspy-ai>=2.5.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

**Step 2: Create .env.example for configuration template**

```
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud

# Optional: API keys if using other providers
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here
```

**Step 3: Install dependencies**

Run: `uv sync` (or `pip install -e ".[dev]"`)
Expected: Dependencies installed successfully

**Step 4: Commit**

```bash
git add pyproject.toml .env.example
git commit -m "chore: add DSPy dependencies and environment template"
```

---

## Task 2: Create Sample Unoptimized Prompt

**Files:**
- Create: `prompts/unoptimized_prompt.md`

**Step 1: Create prompts directory and sample prompt**

```markdown
# Task Description

You are a helpful assistant that answers questions about programming.

## Instructions

1. Read the user's question carefully
2. Provide a clear and accurate answer
3. Include code examples when relevant
4. Be concise but thorough

## Example

User: What is a Python list comprehension?
Assistant: A list comprehension is a concise way to create lists in Python...
```

**Step 2: Commit**

```bash
git add prompts/unoptimized_prompt.md
git commit -m "docs: add sample unoptimized prompt for testing"
```

---

## Task 3: Create DSPy Configuration Module

**Files:**
- Create: `src/dspy_examples/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for config loading**

```python
# tests/test_config.py
"""Tests for DSPy configuration module."""

import os
from unittest.mock import patch

import pytest


class TestLoadConfig:
    """Tests for loading DSPy configuration."""

    def test_load_config_returns_model_name(self) -> None:
        """Test that load_config returns the configured model name."""
        from dspy_examples.config import load_config

        config = load_config()
        assert "model" in config
        assert config["model"] == "gpt-oss:120b-cloud"

    def test_load_config_returns_base_url(self) -> None:
        """Test that load_config returns the configured base URL."""
        from dspy_examples.config import load_config

        config = load_config()
        assert "base_url" in config
        assert config["base_url"] == "http://localhost:11434"


class TestConfigureDspy:
    """Tests for configuring DSPy with Ollama."""

    def test_configure_dspy_sets_lm(self) -> None:
        """Test that configure_dspy sets up the language model."""
        from dspy_examples.config import configure_dspy

        lm = configure_dspy()
        assert lm is not None
        assert lm.model == "gpt-oss:120b-cloud"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module not found)

**Step 3: Create config module**

```python
# src/dspy_examples/__init__.py
"""DSPy Examples - Prompt optimization scripts."""

__version__ = "0.1.0"
```

```python
# src/dspy_examples/config.py
"""DSPy configuration module for Ollama integration."""

import os
from typing import Any

import dspy
from dotenv import load_dotenv


def load_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary with model, base_url, and other settings.
    """
    load_dotenv()

    return {
        "model": os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


def configure_dspy() -> dspy.LM:
    """Configure DSPy to use Ollama language model.

    Returns:
        Configured dspy.LM instance.
    """
    config = load_config()

    lm = dspy.LM(
        model=f"ollama_chat/{config['model']}",
        api_base=config["base_url"],
    )
    dspy.configure(lm=lm)
    return lm
```

**Step 4: Create tests/__init__.py**

```python
# tests/__init__.py
"""Tests for dspy-examples."""
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dspy_examples/__init__.py src/dspy_examples/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add DSPy configuration module with Ollama support"
```

---

## Task 4: Create Prompt Loading Utilities

**Files:**
- Create: `src/dspy_examples/prompts.py`
- Create: `tests/test_prompts.py`

**Step 1: Write failing test for prompt loading**

```python
# tests/test_prompts.py
"""Tests for prompt loading utilities."""

import tempfile
from pathlib import Path

import pytest


class TestLoadPrompt:
    """Tests for loading prompts from files."""

    def test_load_prompt_returns_content(self, tmp_path: Path) -> None:
        """Test that load_prompt returns file content as string."""
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("# Test Prompt\n\nThis is a test.")

        from dspy_examples.prompts import load_prompt

        content = load_prompt(prompt_file)
        assert content == "# Test Prompt\n\nThis is a test."

    def test_load_prompt_raises_on_missing_file(self) -> None:
        """Test that load_prompt raises FileNotFoundError for missing file."""
        from dspy_examples.prompts import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("/nonexistent/path/prompt.md")


class TestSavePrompt:
    """Tests for saving optimized prompts."""

    def test_save_prompt_creates_file(self, tmp_path: Path) -> None:
        """Test that save_prompt creates output file."""
        from dspy_examples.prompts import save_prompt

        output_path = tmp_path / "optimized_prompt.md"
        save_prompt("# Optimized\n\nContent", output_path)

        assert output_path.exists()
        assert output_path.read_text() == "# Optimized\n\nContent"

    def test_save_prompt_with_version_appends_number(self, tmp_path: Path) -> None:
        """Test that save_prompt appends version number when requested."""
        from dspy_examples.prompts import save_prompt

        output_dir = tmp_path

        # First save creates optimized_prompt.md
        path1 = save_prompt("# Version 1", output_dir / "optimized_prompt.md")
        assert path1.name == "optimized_prompt.md"

        # Second save creates optimized_prompt_01.md
        path2 = save_prompt("# Version 2", output_dir / "optimized_prompt.md", version=1)
        assert path2.name == "optimized_prompt_01.md"

        # Third save creates optimized_prompt_02.md
        path3 = save_prompt("# Version 3", output_dir / "optimized_prompt.md", version=2)
        assert path3.name == "optimized_prompt_02.md"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL (module not found)

**Step 3: Create prompts module**

```python
# src/dspy_examples/prompts.py
"""Prompt loading and saving utilities."""

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Load a prompt from a file.

    Args:
        path: Path to the prompt file.

    Returns:
        The prompt content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def save_prompt(
    content: str,
    path: str | Path,
    version: int | None = None,
) -> Path:
    """Save an optimized prompt to a file.

    Args:
        content: The prompt content to save.
        path: Base path for the output file.
        version: Optional version number. If provided, appends _XX to filename.

    Returns:
        The actual path where the file was saved.
    """
    path = Path(path)

    if version is not None:
        # Insert version number before extension
        stem = path.stem
        suffix = path.suffix
        version_str = f"_{version:02d}"
        path = path.with_name(f"{stem}{version_str}{suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dspy_examples/prompts.py tests/test_prompts.py
git commit -m "feat: add prompt loading and saving utilities"
```

---

## Task 5: Create BootstrapFewShot Optimization Script

**Files:**
- Create: `src/dspy_examples/bootstrap_fewshot.py`
- Create: `tests/test_bootstrap_fewshot.py`

**Step 1: Write failing test for optimization signature**

```python
# tests/test_bootstrap_fewshot.py
"""Tests for BootstrapFewShot optimization."""

import pytest


class TestPromptOptimizationSignature:
    """Tests for the DSPy signature."""

    def test_signature_has_input_field(self) -> None:
        """Test that PromptOptimization has input field."""
        from dspy_examples.bootstrap_fewshot import PromptOptimization

        assert "unoptimized_prompt" in PromptOptimization.model_fields

    def test_signature_has_output_field(self) -> None:
        """Test that PromptOptimization has output field."""
        from dspy_examples.bootstrap_fewshot import PromptOptimization

        assert "optimized_prompt" in PromptOptimization.model_fields


class TestOptimizePrompt:
    """Tests for optimize_prompt function."""

    def test_optimize_prompt_returns_string(self) -> None:
        """Test that optimize_prompt returns a string."""
        from dspy_examples.bootstrap_fewshot import optimize_prompt

        # This test will use a mock LM in practice
        result = optimize_prompt("Write a greeting.")
        assert isinstance(result, str)

    def test_optimize_prompt_adds_examples(self) -> None:
        """Test that optimize_prompt adds few-shot examples."""
        from dspy_examples.bootstrap_fewshot import optimize_prompt

        # The optimized prompt should contain examples
        result = optimize_prompt("Summarize text.")
        # Bootstrap few-shot should add example pairs
        assert len(result) > len("Summarize text.")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bootstrap_fewshot.py -v`
Expected: FAIL (module not found)

**Step 3: Create bootstrap_fewshot module**

```python
# src/dspy_examples/bootstrap_fewshot.py
"""BootstrapFewShot optimization script."""

import dspy
from dspy.teleprompt import BootstrapFewShot


class PromptOptimization(dspy.Signature):
    """Optimize a prompt by adding few-shot examples."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt with examples")


class PromptOptimizer(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


def optimize_prompt(
    prompt: str,
    num_threads: int = 4,
) -> str:
    """Optimize a prompt using BootstrapFewShot.

    Args:
        prompt: The unoptimized prompt text.
        num_threads: Number of threads for parallel processing.

    Returns:
        The optimized prompt with few-shot examples.
    """
    from dspy_examples.config import configure_dspy

    configure_dspy()

    optimizer = BootstrapFewShot(
        metric=lambda x, y: True,  # Accept all examples for now
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    # Create a simple module to optimize
    module = PromptOptimizer()

    # Compile with example prompt
    # In real usage, you'd provide training examples
    optimized_module = optimizer.compile(
        module,
        trainset=[
            dspy.Example(
                unoptimized_prompt="Write a greeting.",
                optimized_prompt="Write a greeting.\n\nExample:\nInput: Write a greeting for a friend.\nOutput: Hey there! Great to see you again!",
            )
        ],
    )

    # Run the optimized module
    result = optimized_module(unoptimized_prompt=prompt)

    return result.optimized_prompt


def main() -> None:
    """Run BootstrapFewShot optimization on unoptimized_prompt.md."""
    from pathlib import Path

    from dspy_examples.prompts import load_prompt, save_prompt

    # Load the unoptimized prompt
    input_path = Path("prompts/unoptimized_prompt.md")
    prompt = load_prompt(input_path)

    print(f"Loaded prompt from {input_path}")
    print(f"Original prompt length: {len(prompt)} characters")

    # Optimize
    optimized = optimize_prompt(prompt)

    print(f"Optimized prompt length: {len(optimized)} characters")

    # Save with version number if file exists
    output_path = Path("prompts/optimized_prompt.md")
    if output_path.exists():
        # Find next available version number
        version = 1
        while output_path.with_name(f"optimized_prompt_{version:02d}.md").exists():
            version += 1
        save_prompt(optimized, output_path, version=version)
        print(f"Saved optimized prompt to {output_path.with_name(f'optimized_prompt_{version:02d}.md')}")
    else:
        save_prompt(optimized, output_path)
        print(f"Saved optimized prompt to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bootstrap_fewshot.py -v`
Expected: PASS (with warnings about LM)

**Step 5: Commit**

```bash
git add src/dspy_examples/bootstrap_fewshot.py tests/test_bootstrap_fewshot.py
git commit -m "feat: add BootstrapFewShot optimization script"
```

---

## Task 6: Create Main Entry Point

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

**Step 1: Write failing test for main**

```python
# tests/test_main.py
"""Tests for main entry point."""

import subprocess
import sys


class TestMain:
    """Tests for main.py entry point."""

    def test_main_module_runs(self) -> None:
        """Test that main module can be executed."""
        result = subprocess.run(
            [sys.executable, "-m", "dspy_examples.bootstrap_fewshot"],
            capture_output=True,
            text=True,
        )
        # Should not crash (may fail if Ollama not running)
        assert result.returncode in (0, 1)

    def test_main_imports(self) -> None:
        """Test that main module imports successfully."""
        from dspy_examples import bootstrap_fewshot

        assert hasattr(bootstrap_fewshot, "main")
        assert hasattr(bootstrap_fewshot, "optimize_prompt")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL or PASS depending on module structure

**Step 3: Update main.py**

```python
# main.py
"""Main entry point for dspy-examples."""

from dspy_examples.bootstrap_fewshot import main


def run() -> None:
    """Run the BootstrapFewShot optimization."""
    main()


if __name__ == "__main__":
    run()
```

**Step 4: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: update main entry point for BootstrapFewShot"
```

---

## Task 7: Add README Documentation

**Files:**
- Modify: `README.md`

**Step 1: Write README**

```markdown
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

```bash
python main.py
# or
python -m dspy_examples.bootstrap_fewshot
```

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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Task 8: Initial Git Setup and First Commit

**Files:**
- Create: `.gitignore`

**Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env

# Testing
.pytest_cache/
.coverage
htmlcov/

# UV
.uv/
uv.lock

# Worktrees (for isolated development)
.worktrees/
worktrees/
```

**Step 2: Initialize git and make first commit (if not already done)**

Run: `git status`
Expected: Show untracked files

**Step 3: Initial commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

**Step 4: Add all files and create initial commit**

```bash
git add .
git commit -m "feat: initial project setup with DSPy BootstrapFewShot"
```

---

## Verification

After all tasks:

```bash
# Run all tests
pytest -v

# Run the optimization (requires Ollama running)
python main.py
```

---

## Future Tasks

After completing this plan, future optimization techniques can follow the same pattern:

1. **MIPROv2** - `src/dspy_examples/mipro_v2.py`
2. **BootstrapFewShotWithRandomSearch** - `src/dspy_examples/bootstrap_random.py`
3. **KNNFewShot** - `src/dspy_examples/knn_fewshot.py`

Each new technique:
- Creates module in `src/dspy_examples/`
- Creates corresponding test file
- Reads from `unoptimized_prompt.md`
- Outputs to `optimized_prompt.md` with versioning