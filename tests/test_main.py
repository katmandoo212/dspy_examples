# tests/test_main.py
"""Tests for main entry point."""

import subprocess
import sys


class TestMain:
    """Tests for main.py entry point."""

    def test_main_module_imports(self) -> None:
        """Test that main module imports successfully."""
        from dspy_examples import (
            get_settings,
            OptimizationPipeline,
            PipelineConfig,
            ProviderFactory,
            OptimizerFactory,
        )

        assert callable(get_settings)
        assert OptimizationPipeline is not None
        assert PipelineConfig is not None
        assert ProviderFactory is not None
        assert OptimizerFactory is not None

    def test_main_has_main_function(self) -> None:
        """Test that main.py has main function."""
        from main import main

        assert callable(main)