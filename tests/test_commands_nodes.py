"""Tests for PocketFlow nodes for optimization."""

import pytest
from pathlib import Path
import tempfile


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


class TestSaveResultNode:
    """Tests for SaveResultNode."""

    def test_save_result_node(self):
        """Test saving optimization result."""
        from dspy_examples.commands.nodes import SaveResultNode
        from dspy_examples.optimizers.base import OptimizationResult

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "optimized.md"

            node = SaveResultNode(output_path=output_file)

            result = OptimizationResult(
                optimized_prompt="Optimized content here",
                optimizer_name="test",
                provider_name="test",
                original_length=10,
                optimized_length=20,
            )

            shared = {"result": result, "output_path": output_file}
            node.run(shared)

            assert output_file.exists()
            assert output_file.read_text() == "Optimized content here"