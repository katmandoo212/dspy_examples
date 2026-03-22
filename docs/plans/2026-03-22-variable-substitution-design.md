# Variable Substitution in Prompts - Design Document

**Date**: 2026-03-22
**Status**: Approved for Implementation

## Summary

Add variable substitution support to prompts with configurable delimiters, preserve/substitute modes, and integration at both optimization-time and inference-time.

## Requirements

### Functional Requirements

1. **Variable Syntax**: Configurable delimiters (default `[[ ]]`) identify variables in prompts
2. **Variable Modes**: Each variable has `preserve` or `substitute` mode
   - `preserve`: Keep placeholder in optimized output
   - `substitute`: Replace with value, optimizer may generalize
3. **Configuration**: Project-wide default delimiter, per-prompt override
4. **Two-Phase Substitution**:
   - Optimization-time: Defaults from metadata + training example values
   - Inference-time: Pipeline arguments only
5. **Backwards Compatibility**: Prompts without frontmatter work unchanged

### Non-Functional Requirements

- Clean API surface
- Comprehensive test coverage
- Minimal changes to existing pipeline

## File Format

### YAML Frontmatter

```markdown
---
# Delimiter configuration (optional, uses project default if omitted)
delimiter: "[[ ]]"

# Variable definitions
variables:
  country:
    mode: preserve
    default: France
    description: The country to query about

  tone:
    mode: substitute
    default: formal
    description: Response tone style
---

# Task Description

Answer questions about [[country]] in a [[tone]] manner.
```

### Parsing Rules

1. Frontmatter is YAML between `---` delimiters at file start
2. `delimiter` can be:
   - String like `"[[ ]]"` (split on space → start/end)
   - Object with `start` and `end` keys
   - Omitted (uses project default)
3. `variables` maps variable name to definition with `mode`, `default`, `description`
4. Body content follows second `---` (or entire file if no frontmatter)
5. Variables found in content but not in frontmatter: auto-detected, default to `mode: preserve`

## Architecture

### Data Structures

```python
@dataclass
class VariableDef:
    """Definition of a variable in a prompt."""
    name: str
    mode: Literal["preserve", "substitute"]
    default: str | None = None
    description: str | None = None

@dataclass
class DelimiterConfig:
    """Delimiter configuration for variable detection."""
    start: str = "[["
    end: str = "]]"

@dataclass
class ParsedPrompt:
    """Result of parsing a prompt file."""
    content: str                    # Raw content without frontmatter
    variables: dict[str, VariableDef]
    delimiter: DelimiterConfig
    metadata: dict[str, Any]        # Any other frontmatter fields
```

### PromptTemplate Class

```python
class PromptTemplate:
    """Template for prompts with variable substitution."""

    def __init__(self, parsed: ParsedPrompt):
        self._parsed = parsed

    @classmethod
    def from_file(
        cls,
        path: Path,
        default_delimiter: DelimiterConfig | None = None
    ) -> "PromptTemplate":
        """Load and parse a prompt file."""
        ...

    @classmethod
    def from_string(
        cls,
        content: str,
        default_delimiter: DelimiterConfig | None = None
    ) -> "PromptTemplate":
        """Parse prompt from string."""
        ...

    def substitute(self, values: dict[str, str]) -> str:
        """Substitute variables with values, respecting modes.

        - preserve mode: replaces placeholder with value but tracks for restoration
        - substitute mode: replaces placeholder with value permanently
        """
        ...

    def validate(self, values: dict[str, str] | None = None) -> list[str]:
        """Return list of validation errors."""
        ...

    def get_preserved_variables(self) -> list[str]:
        """Return variables that should be preserved after substitution."""
        ...

    def extract_variables(self) -> list[str]:
        """Find all variables in content using current delimiter."""
        ...

    @property
    def variables(self) -> dict[str, VariableDef]:
        return self._parsed.variables

    @property
    def content(self) -> str:
        return self._parsed.content
```

### Settings Integration

```python
# settings.py
class Settings(BaseSettings):
    # ... existing settings ...

    # Variable substitution defaults
    variable_delimiter_start: str = "[["
    variable_delimiter_end: str = "]]"

    def get_default_delimiter(self) -> DelimiterConfig:
        return DelimiterConfig(
            start=self.variable_delimiter_start,
            end=self.variable_delimiter_end
        )
```

### Pipeline Integration

```python
# pipeline.py
class OptimizationPipeline:
    def run(
        self,
        variables: dict[str, str] | None = None,
        **kwargs
    ) -> OptimizationResult:
        # Load template
        template = PromptTemplate.from_file(
            self.config.input_path,
            default_delimiter=self.settings.get_default_delimiter()
        )

        # Validate
        errors = template.validate(variables)
        if errors:
            raise ValueError(f"Variable validation failed: {errors}")

        # Substitute for optimization
        prompt = template.substitute(variables or {})

        # Run optimizer
        result = optimizer.optimize(prompt, trainset)

        # Restore preserved placeholders
        if template.has_preserved_variables():
            result.optimized_prompt = self._restore_placeholders(
                result.optimized_prompt,
                template.get_preserved_variables()
            )

        return result
```

### Training Example Variables

```python
# Training examples can provide variable values
trainset = [
    dspy.Example(
        unoptimized_prompt="What is the capital of [[country]]?",
        optimized_prompt="What is the capital of France?...",
        variables={"country": "France", "tone": "formal"}
    ).with_inputs("unoptimized_prompt")
]
```

## File Organization

```
src/dspy_examples/
├── template.py              # NEW: PromptTemplate, VariableDef, DelimiterConfig
├── prompts.py               # MODIFIED: Add from_template() helper
├── settings.py              # MODIFIED: Add delimiter defaults
└── pipeline.py              # MODIFIED: Accept variables parameter

tests/
├── test_template.py         # NEW: PromptTemplate unit tests
├── test_template_integration.py  # NEW: Pipeline integration tests
└── fixtures/prompts/        # NEW: Test prompt files
    ├── simple.md
    ├── no_frontmatter.md
    ├── preserve_mode.md
    └── custom_delimiter.md
```

## Test Cases

### Unit Tests (test_template.py)

1. `test_parse_frontmatter` - Correctly parse YAML frontmatter
2. `test_parse_no_frontmatter` - Handle plain prompts without frontmatter
3. `test_extract_variables` - Find all `[[var]]` patterns in content
4. `test_custom_delimiter` - Parse with non-default delimiter
5. `test_substitute_all` - Replace all placeholders with values
6. `test_substitute_preserve_mode` - Track preserved variables for restoration
7. `test_substitute_substitute_mode` - Permanent replacement
8. `test_validate_missing_required` - Error when required value missing
9. `test_validate_extra_values` - Warn but don't error on extra values
10. `test_default_values` - Use defaults when values not provided

### Integration Tests (test_template_integration.py)

1. `test_pipeline_no_variables` - Backwards compatibility
2. `test_pipeline_with_variables` - End-to-end substitution
3. `test_training_example_variables` - Training values override defaults
4. `test_preserved_in_output` - Verify placeholders restored in optimized output

## Backwards Compatibility

- Prompts without frontmatter work unchanged
- Existing pipeline calls without `variables` parameter work unchanged
- Default delimiter `[[ ]]` is safe for most prompts

## Implementation Order

1. Create `template.py` with core classes
2. Add unit tests for `PromptTemplate`
3. Update `settings.py` with delimiter defaults
4. Modify `pipeline.py` to accept `variables` parameter
5. Add integration tests
6. Update `README.md` with usage examples

## Usage Examples

### Basic Usage

```python
from dspy_examples.template import PromptTemplate
from dspy_examples.pipeline import OptimizationPipeline

# Load template
template = PromptTemplate.from_file("prompts/question.md")

# Validate and substitute
values = {"country": "France", "tone": "formal"}
errors = template.validate(values)
if errors:
    print(f"Missing: {errors}")

prompt = template.substitute(values)

# Run pipeline
pipeline = OptimizationPipeline()
result = pipeline.run(variables=values)
```

### Prompt File

```markdown
---
delimiter: "[[ ]]"
variables:
  country:
    mode: preserve
    default: USA
  tone:
    mode: substitute
    default: professional
---

You are a geography expert. Answer questions about [[country]] in a [[tone]] manner.
```

### Inference-Time

```python
# User provides values at runtime
result = pipeline.run(variables={"country": "Japan", "tone": "casual"})

# Preserved variables appear in output:
# "You are a geography expert. Answer questions about [[country]] in a casual manner."
# (country preserved, tone substituted)
```