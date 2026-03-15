# tests/test_main.py
"""Tests for main entry point."""

import subprocess
import sys


class TestMain:
    """Tests for main.py entry point."""

    def test_main_module_imports(self) -> None:
        """Test that main module imports successfully."""
        from dspy_examples import bootstrap_fewshot

        assert hasattr(bootstrap_fewshot, "main")
        assert hasattr(bootstrap_fewshot, "optimize_prompt")

    def test_main_has_run_function(self) -> None:
        """Test that main.py has run function."""
        from main import run

        assert callable(run)