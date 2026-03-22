"""Prompt template with variable substitution support.

This module provides variable substitution in prompts with:
- Configurable delimiters (default [[ ]])
- Preserve/substitute modes per variable
- YAML frontmatter for variable definitions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class VariableDef:
    """Definition of a variable in a prompt.

    Attributes:
        name: Variable identifier (e.g., "country")
        mode: "preserve" (keep in output) or "substitute" (replace permanently)
        default: Default value if none provided
        description: Human-readable description
    """

    name: str
    mode: Literal["preserve", "substitute"] = "preserve"
    default: str | None = None
    description: str | None = None


@dataclass
class DelimiterConfig:
    """Delimiter configuration for variable detection.

    Attributes:
        start: Opening delimiter (default "[[")
        end: Closing delimiter (default "]]")
    """

    start: str = "[["
    end: str = "]]"

    def __post_init__(self) -> None:
        """Validate delimiters are not empty."""
        if not self.start or not self.end:
            raise ValueError("Delimiters cannot be empty")

    @classmethod
    def from_string(cls, delimiter_str: str) -> "DelimiterConfig":
        """Parse delimiter from string like '[[ ]]' or custom format.

        Args:
            delimiter_str: String with space-separated start/end, or YAML-style object.

        Returns:
            DelimiterConfig instance.

        Raises:
            ValueError: If delimiter string is invalid.
        """
        delimiter_str = delimiter_str.strip()

        # Handle "start end" format (e.g., "[[ ]]" or "${ }")
        if " " in delimiter_str:
            parts = delimiter_str.split(None, 1)
            if len(parts) == 2:
                return cls(start=parts[0], end=parts[1])

        # Single delimiter pair like "{}" - split in half
        if len(delimiter_str) >= 2 and delimiter_str[0] != " ":
            mid = len(delimiter_str) // 2
            return cls(start=delimiter_str[:mid], end=delimiter_str[mid:])

        # Fallback: use for both start and end
        return cls(start=delimiter_str, end=delimiter_str)


@dataclass
class ParsedPrompt:
    """Result of parsing a prompt file.

    Attributes:
        content: Raw content without frontmatter
        variables: Variable definitions from frontmatter
        delimiter: Delimiter configuration
        metadata: Other frontmatter fields
    """

    content: str
    variables: dict[str, VariableDef] = field(default_factory=dict)
    delimiter: DelimiterConfig = field(default_factory=DelimiterConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptTemplate:
    """Template for prompts with variable substitution.

    Supports YAML frontmatter for variable definitions and configurable
    delimiters for variable placeholders.

    Example:
        >>> template = PromptTemplate.from_file("prompt.md")
        >>> prompt = template.substitute({"country": "France"})
    """

    # Regex pattern for YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$",
        re.DOTALL
    )

    def __init__(self, parsed: ParsedPrompt) -> None:
        """Initialize from parsed prompt.

        Args:
            parsed: ParsedPrompt containing content, variables, and delimiter.
        """
        self._parsed = parsed
        self._extracted_variables: list[str] | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        default_delimiter: DelimiterConfig | None = None,
    ) -> "PromptTemplate":
        """Load and parse a prompt file.

        Args:
            path: Path to the prompt file.
            default_delimiter: Default delimiter if not specified in file.

        Returns:
            PromptTemplate instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If frontmatter is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        content = path.read_text(encoding="utf-8")
        return cls.from_string(content, default_delimiter)

    @classmethod
    def from_string(
        cls,
        content: str,
        default_delimiter: DelimiterConfig | None = None,
    ) -> "PromptTemplate":
        """Parse prompt from string.

        Args:
            content: Raw prompt content with optional frontmatter.
            default_delimiter: Default delimiter if not specified in content.

        Returns:
            PromptTemplate instance.
        """
        default_delimiter = default_delimiter or DelimiterConfig()

        # Parse frontmatter
        frontmatter: dict[str, Any] = {}
        body = content

        match = cls.FRONTMATTER_PATTERN.match(content)
        if match:
            import yaml
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML frontmatter: {e}") from e
            body = match.group(2)

        # Extract delimiter config
        delimiter = default_delimiter
        if "delimiter" in frontmatter:
            delimiter = DelimiterConfig.from_string(frontmatter["delimiter"])

        # Extract variable definitions
        variables: dict[str, VariableDef] = {}
        var_defs = frontmatter.get("variables", {})
        if isinstance(var_defs, dict):
            for var_name, var_config in var_defs.items():
                if isinstance(var_config, dict):
                    variables[var_name] = VariableDef(
                        name=var_name,
                        mode=var_config.get("mode", "preserve"),
                        default=var_config.get("default"),
                        description=var_config.get("description"),
                    )
                elif isinstance(var_config, str):
                    # Shorthand: just the default value
                    variables[var_name] = VariableDef(
                        name=var_name,
                        default=var_config,
                    )

        # Collect other metadata
        metadata = {
            k: v for k, v in frontmatter.items()
            if k not in ("delimiter", "variables")
        }

        parsed = ParsedPrompt(
            content=body,
            variables=variables,
            delimiter=delimiter,
            metadata=metadata,
        )

        return cls(parsed)

    def _build_pattern(self) -> re.Pattern:
        """Build regex pattern for variable detection."""
        start = re.escape(self._parsed.delimiter.start)
        end = re.escape(self._parsed.delimiter.end)
        pattern = rf"{start}(\w+){end}"
        return re.compile(pattern)

    def extract_variables(self) -> list[str]:
        """Find all variables in content using current delimiter.

        Returns:
            List of variable names found in content.
        """
        if self._extracted_variables is not None:
            return self._extracted_variables

        pattern = self._build_pattern()
        matches = pattern.findall(self._parsed.content)
        self._extracted_variables = list(dict.fromkeys(matches))  # Preserve order, unique
        return self._extracted_variables

    def get_all_variables(self) -> dict[str, VariableDef]:
        """Get all variables (defined + discovered from content).

        Variables found in content but not in frontmatter get default config.
        """
        all_vars = dict(self._parsed.variables)

        # Add discovered variables not in frontmatter
        for var_name in self.extract_variables():
            if var_name not in all_vars:
                all_vars[var_name] = VariableDef(
                    name=var_name,
                    mode="preserve",  # Default to preserve
                )

        return all_vars

    def substitute(self, values: dict[str, str]) -> str:
        """Substitute variables with values, respecting modes.

        Args:
            values: Mapping of variable names to values.

        Returns:
            Content with variables substituted.

        Note:
            - preserve mode: Substitutes value but tracks for restoration
            - substitute mode: Permanently replaces placeholder
        """
        result = self._parsed.content
        pattern = self._build_pattern()
        all_vars = self.get_all_variables()

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            var_def = all_vars.get(var_name)

            # Get value: provided > default > keep placeholder
            if var_name in values:
                return values[var_name]
            if var_def and var_def.default is not None:
                return var_def.default

            # No value available, keep placeholder
            return match.group(0)

        return pattern.sub(replace_var, result)

    def validate(self, values: dict[str, str] | None = None) -> list[str]:
        """Validate that all required variables have values.

        Args:
            values: Provided values (or None to check defaults).

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        values = values or {}
        all_vars = self.get_all_variables()

        for var_name in self.extract_variables():
            var_def = all_vars.get(var_name)
            has_value = var_name in values
            has_default = var_def and var_def.default is not None

            if not has_value and not has_default:
                errors.append(f"Missing value for variable '{var_name}' (no default provided)")

        # Warn about extra values (not an error, but informative)
        for var_name in values:
            if var_name not in all_vars:
                errors.append(f"Warning: Unknown variable '{var_name}' provided")

        return errors

    def get_preserved_variables(self) -> list[str]:
        """Return variables that should be preserved after substitution.

        Returns:
            List of variable names with mode='preserve'.
        """
        all_vars = self.get_all_variables()
        return [name for name, var in all_vars.items() if var.mode == "preserve"]

    def has_preserved_variables(self) -> bool:
        """Check if any variables should be preserved in output."""
        return bool(self.get_preserved_variables())

    @property
    def variables(self) -> dict[str, VariableDef]:
        """Get defined variables from frontmatter."""
        return self._parsed.variables

    @property
    def content(self) -> str:
        """Get raw content without frontmatter."""
        return self._parsed.content

    @property
    def delimiter(self) -> DelimiterConfig:
        """Get delimiter configuration."""
        return self._parsed.delimiter

    @property
    def metadata(self) -> dict[str, Any]:
        """Get additional frontmatter metadata."""
        return self._parsed.metadata