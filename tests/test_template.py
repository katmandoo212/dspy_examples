"""Tests for PromptTemplate and variable substitution."""

import pytest
from pathlib import Path

from dspy_examples.template import (
    PromptTemplate,
    VariableDef,
    DelimiterConfig,
    ParsedPrompt,
)


# Fixtures directory
FIXTURES = Path(__file__).parent / "fixtures" / "prompts"


class TestDelimiterConfig:
    """Tests for DelimiterConfig."""

    def test_default_delimiters(self):
        """Test default delimiter configuration."""
        config = DelimiterConfig()
        assert config.start == "[["
        assert config.end == "]]"

    def test_custom_delimiters(self):
        """Test custom delimiter configuration."""
        config = DelimiterConfig(start="${", end="}")
        assert config.start == "${"
        assert config.end == "}"

    def test_from_string_space_separated(self):
        """Test parsing space-separated delimiter string."""
        config = DelimiterConfig.from_string("[[ ]]")
        assert config.start == "[["
        assert config.end == "]]"

    def test_from_string_single_delimiter(self):
        """Test parsing single delimiter pair like '{}'."""
        config = DelimiterConfig.from_string("{}")
        assert config.start == "{"
        assert config.end == "}"

    def test_empty_delimiter_raises(self):
        """Test that empty delimiters raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DelimiterConfig(start="", end="]]")


class TestVariableDef:
    """Tests for VariableDef."""

    def test_default_mode(self):
        """Test default mode is preserve."""
        var = VariableDef(name="country")
        assert var.mode == "preserve"
        assert var.default is None
        assert var.description is None

    def test_custom_values(self):
        """Test custom variable definition."""
        var = VariableDef(
            name="country",
            mode="substitute",
            default="France",
            description="The country to query"
        )
        assert var.mode == "substitute"
        assert var.default == "France"
        assert var.description == "The country to query"


class TestPromptTemplateParsing:
    """Tests for PromptTemplate parsing."""

    def test_from_string_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "# Simple Prompt\n\nWhat is the capital of [[country]]?"
        template = PromptTemplate.from_string(content)

        assert "What is the capital of [[country]]?" in template.content
        assert template.variables == {}
        assert template.delimiter.start == "[["
        assert template.delimiter.end == "]]"

    def test_from_string_with_frontmatter(self):
        """Test parsing content with YAML frontmatter."""
        content = """---
variables:
  country:
    mode: preserve
    default: France
---

What is the capital of [[country]]?"""
        template = PromptTemplate.from_string(content)

        assert "What is the capital of [[country]]?" in template.content
        assert "country" in template.variables
        assert template.variables["country"].mode == "preserve"
        assert template.variables["country"].default == "France"

    def test_from_string_custom_delimiter(self):
        """Test parsing with custom delimiter."""
        content = """---
delimiter: "${ }"
variables:
  topic: science
---

Discuss ${topic} in detail."""
        template = PromptTemplate.from_string(content)

        assert template.delimiter.start == "${"
        assert template.delimiter.end == "}"
        assert template.variables["topic"].default == "science"

    def test_from_string_shorthand_variable(self):
        """Test shorthand variable definition (just default value)."""
        content = """---
variables:
  country: France
  tone: formal
---

Answer about [[country]] in a [[tone]] way."""
        template = PromptTemplate.from_string(content)

        assert template.variables["country"].default == "France"
        assert template.variables["country"].mode == "preserve"
        assert template.variables["tone"].default == "formal"

    def test_from_string_metadata(self):
        """Test that additional frontmatter fields are preserved."""
        content = """---
title: My Prompt
version: "1.0"
author: Test
---

Content here."""
        template = PromptTemplate.from_string(content)

        assert template.metadata["title"] == "My Prompt"
        assert template.metadata["version"] == "1.0"
        assert template.metadata["author"] == "Test"

    def test_from_file(self):
        """Test loading template from file."""
        template = PromptTemplate.from_file(FIXTURES / "simple.md")

        assert "Answer questions about [[country]]" in template.content
        assert "tone" in template.extract_variables()

    def test_from_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file("/nonexistent/path/file.md")

    def test_from_file_no_frontmatter(self):
        """Test loading file without frontmatter."""
        template = PromptTemplate.from_file(FIXTURES / "no_frontmatter.md")

        assert "What is the capital of France" in template.content
        assert len(template.variables) == 0

    def test_from_file_with_preserve_mode(self):
        """Test loading file with preserve mode variables."""
        template = PromptTemplate.from_file(FIXTURES / "preserve_mode.md")

        assert "country" in template.variables
        assert template.variables["country"].mode == "preserve"
        assert template.variables["country"].default == "USA"
        assert template.variables["tone"].mode == "substitute"

    def test_from_file_custom_delimiter(self):
        """Test loading file with custom delimiter."""
        template = PromptTemplate.from_file(
            FIXTURES / "custom_delimiter.md",
            default_delimiter=DelimiterConfig(start="[[", end="]]")
        )

        assert template.delimiter.start == "${"
        assert template.delimiter.end == "}"

    def test_default_delimiter_applied(self):
        """Test that default delimiter is used when not in frontmatter."""
        content = "What is [[topic]]?"
        template = PromptTemplate.from_string(
            content,
            default_delimiter=DelimiterConfig(start="${", end="}")
        )

        assert template.delimiter.start == "${"
        assert template.delimiter.end == "}"


class TestPromptTemplateExtraction:
    """Tests for variable extraction."""

    def test_extract_variables_basic(self):
        """Test extracting variables from content."""
        content = "What is the capital of [[country]]? Discuss [[topic]]."
        template = PromptTemplate.from_string(content)

        vars_found = template.extract_variables()
        assert "country" in vars_found
        assert "topic" in vars_found

    def test_extract_variables_duplicates(self):
        """Test that duplicate variables are deduplicated."""
        content = "[[country]] is great. [[country]] has [[city]]."
        template = PromptTemplate.from_string(content)

        vars_found = template.extract_variables()
        assert vars_found.count("country") == 1
        assert "city" in vars_found

    def test_extract_variables_custom_delimiter(self):
        """Test extraction with custom delimiter."""
        content = "Discuss ${topic} and ${subject}."
        template = PromptTemplate.from_string(
            content,
            default_delimiter=DelimiterConfig(start="${", end="}")
        )

        vars_found = template.extract_variables()
        assert "topic" in vars_found
        assert "subject" in vars_found

    def test_get_all_variables(self):
        """Test combining defined and discovered variables."""
        content = """---
variables:
  country:
    mode: preserve
    default: France
---

[[country]] and [[city]] are places."""
        template = PromptTemplate.from_string(content)

        all_vars = template.get_all_variables()

        assert "country" in all_vars
        assert "city" in all_vars
        assert all_vars["country"].default == "France"
        assert all_vars["city"].mode == "preserve"  # Default for discovered


class TestPromptTemplateSubstitution:
    """Tests for variable substitution."""

    def test_substitute_all_values(self):
        """Test substituting all variables."""
        content = "[[country]] has capital [[city]]."
        template = PromptTemplate.from_string(content)

        result = template.substitute({"country": "France", "city": "Paris"})
        assert result == "France has capital Paris."

    def test_substitute_partial_values(self):
        """Test substituting some variables, keeping others as placeholders."""
        content = "[[country]] has capital [[city]]."
        template = PromptTemplate.from_string(content)

        result = template.substitute({"country": "France"})
        assert result == "France has capital [[city]]."

    def test_substitute_with_defaults(self):
        """Test using defaults when values not provided."""
        content = """---
variables:
  country:
    default: USA
  city:
    default: Washington
---

[[country]] - [[city]]."""
        template = PromptTemplate.from_string(content)

        result = template.substitute({})
        assert result == "USA - Washington."

    def test_substitute_values_override_defaults(self):
        """Test that provided values override defaults."""
        content = """---
variables:
  country:
    default: USA
---

[[country]]."""
        template = PromptTemplate.from_string(content)

        result = template.substitute({"country": "France"})
        assert result == "France."

    def test_substitute_custom_delimiter(self):
        """Test substitution with custom delimiter."""
        content = """---
delimiter: "${ }"
---

${topic} is interesting."""
        template = PromptTemplate.from_string(content)

        result = template.substitute({"topic": "Python"})
        assert result == "Python is interesting."

    def test_substitute_no_variables(self):
        """Test that content without variables is unchanged."""
        content = "Simple text without variables."
        template = PromptTemplate.from_string(content)

        result = template.substitute({})
        assert result == "Simple text without variables."


class TestPromptTemplateValidation:
    """Tests for variable validation."""

    def test_validate_all_present(self):
        """Test validation when all variables have values."""
        content = "[[country]] and [[city]]."
        template = PromptTemplate.from_string(content)

        errors = template.validate({"country": "France", "city": "Paris"})
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation when required values are missing."""
        content = "[[country]] and [[city]]."
        template = PromptTemplate.from_string(content)

        errors = template.validate({})
        assert len(errors) == 2
        assert any("country" in e for e in errors)
        assert any("city" in e for e in errors)

    def test_validate_with_defaults(self):
        """Test validation uses defaults."""
        content = """---
variables:
  country:
    default: USA
  city:
---

[[country]] and [[city]]."""
        template = PromptTemplate.from_string(content)

        errors = template.validate({})
        assert len(errors) == 1
        assert "city" in errors[0]
        assert "country" not in errors[0]  # Has default

    def test_validate_unknown_variable(self):
        """Test warning for unknown variables."""
        content = "[[country]]."
        template = PromptTemplate.from_string(content)

        errors = template.validate({"country": "France", "unknown": "value"})
        assert any("unknown" in e.lower() for e in errors)

    def test_validate_none_values(self):
        """Test validation with None values (check defaults)."""
        content = """---
variables:
  country:
    default: France
---

[[country]]."""
        template = PromptTemplate.from_string(content)

        errors = template.validate(None)
        assert len(errors) == 0  # Default provided


class TestPromptTemplatePreserve:
    """Tests for preserve mode functionality."""

    def test_get_preserved_variables(self):
        """Test getting variables with preserve mode."""
        content = """---
variables:
  country:
    mode: preserve
  tone:
    mode: substitute
---

[[country]] in [[tone]]."""
        template = PromptTemplate.from_string(content)

        preserved = template.get_preserved_variables()
        assert "country" in preserved
        assert "tone" not in preserved

    def test_has_preserved_variables(self):
        """Test checking for preserved variables."""
        content = """---
variables:
  country:
    mode: preserve
---

[[country]]."""
        template = PromptTemplate.from_string(content)

        assert template.has_preserved_variables() is True

    def test_no_preserved_variables(self):
        """Test when no variables are preserved."""
        content = """---
variables:
  country:
    mode: substitute
---

[[country]]."""
        template = PromptTemplate.from_string(content)

        assert template.has_preserved_variables() is False


class TestPromptTemplateProperties:
    """Tests for template properties."""

    def test_variables_property(self):
        """Test variables property returns defined variables."""
        content = """---
variables:
  country: France
---

[[country]] and [[city]]."""
        template = PromptTemplate.from_string(content)

        # Only returns defined variables, not discovered
        assert "country" in template.variables
        assert "city" not in template.variables

    def test_content_property(self):
        """Test content property returns body content."""
        content = """---
variables:
  country: France
---

Body content here."""
        template = PromptTemplate.from_string(content)

        assert template.content == "Body content here."

    def test_delimiter_property(self):
        """Test delimiter property."""
        content = """---
delimiter: "${ }"
---

Content."""
        template = PromptTemplate.from_string(content)

        assert template.delimiter.start == "${"
        assert template.delimiter.end == "}"

    def test_metadata_property(self):
        """Test metadata property returns frontmatter metadata."""
        content = """---
title: Test
custom: value
---

Content."""
        template = PromptTemplate.from_string(content)

        assert template.metadata["title"] == "Test"
        assert template.metadata["custom"] == "value"