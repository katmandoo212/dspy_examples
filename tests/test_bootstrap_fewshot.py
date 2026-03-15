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


class TestPromptOptimizerModule:
    """Tests for the PromptOptimizer module."""

    def test_module_can_be_instantiated(self) -> None:
        """Test that PromptOptimizer can be instantiated."""
        from dspy_examples.bootstrap_fewshot import PromptOptimizer

        module = PromptOptimizer()
        assert module is not None
        assert hasattr(module, "prog")

    def test_module_has_forward_method(self) -> None:
        """Test that PromptOptimizer has forward method."""
        from dspy_examples.bootstrap_fewshot import PromptOptimizer

        module = PromptOptimizer()
        assert hasattr(module, "forward")


class TestOptimizePrompt:
    """Tests for optimize_prompt function.

    Note: optimize_prompt requires a configured language model (Ollama).
    These tests are marked with @pytest.mark.integration and require
    an actual LM to be configured.
    """

    @pytest.mark.integration
    def test_optimize_prompt_returns_string(self) -> None:
        """Test that optimize_prompt returns a string.

        Requires: Ollama running with gpt-oss:120b-cloud model
        """
        pytest.skip("Integration test requires running Ollama with gpt-oss:120b-cloud")

    @pytest.mark.integration
    def test_optimize_prompt_adds_examples(self) -> None:
        """Test that optimize_prompt adds few-shot examples.

        Requires: Ollama running with gpt-oss:120b-cloud model
        """
        pytest.skip("Integration test requires running Ollama with gpt-oss:120b-cloud")