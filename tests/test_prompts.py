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