"""Template pattern for prompt optimization workflow."""

from dataclasses import dataclass, field
from pathlib import Path
import time
import hashlib
from typing import Any

from dspy_examples.cache import OptimizationCache
from dspy_examples.factory.provider_factory import ProviderFactory
from dspy_examples.factory.optimizer_factory import OptimizerFactory
from dspy_examples.optimizers.base import OptimizationResult, OptimizerConfig
from dspy_examples.prompts import load_prompt, save_prompt
from dspy_examples.settings import get_settings
from dspy_examples.template import PromptTemplate
from dspy_examples.observers.observable import Observable


@dataclass
class PipelineConfig:
    """Configuration for the optimization pipeline."""

    provider_name: str | None = None  # None = use settings default
    optimizer_name: str | None = None  # None = use settings default
    input_path: Path = Path("prompts/unoptimized_prompt.md")
    output_path: Path = Path("prompts/optimized_prompt.md")
    use_cache: bool = True
    trainset: list[Any] | None = None  # Training examples
    variables: dict[str, str] | None = None  # Variable values for substitution


class OptimizationPipeline(Observable):
    """Template pattern for prompt optimization workflow."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        self.settings = get_settings()
        self.cache = OptimizationCache() if self.config.use_cache else None
        self._template: PromptTemplate | None = None

    def run(self, variables: dict[str, str] | None = None) -> OptimizationResult:
        """Execute the optimization pipeline.

        Args:
            variables: Variable values for prompt substitution.
                      Merged with any variables from config.
                      If None, uses config.variables.

        Returns:
            OptimizationResult with the optimized prompt.
        """
        # Merge variables from config and runtime
        runtime_vars = self._merge_variables(variables)

        # 1. Setup
        self._setup()

        # 2. Load and process prompt template
        prompt, template = self._load_prompt_with_variables(runtime_vars)

        # 3. Get provider
        provider_name = self.config.provider_name or self.settings.llm_provider
        provider = self._get_provider(provider_name)

        # 4. Check cache
        optimizer_name = self.config.optimizer_name or self.settings.optimizer
        cache_key = self._get_cache_key(prompt, optimizer_name, provider_name)
        if self.cache and (cached := self.cache.get(cache_key)):
            return cached

        # 5. Get optimizer
        optimizer = self._get_optimizer(optimizer_name)

        # 6. Run optimization
        start_time = time.time()
        result = self._optimize(prompt, optimizer)
        result.optimization_time = time.time() - start_time
        result.cache_key = cache_key
        result.provider_name = provider_name

        # 7. Restore preserved variables in output
        if template and template.has_preserved_variables():
            result.optimized_prompt = self._restore_preserved_variables(
                result.optimized_prompt,
                template,
                runtime_vars
            )

        # 8. Cache result
        if self.cache:
            self.cache.set(cache_key, result)

        # 9. Save output
        self._save_result(result)

        # 10. Cleanup
        self._cleanup()

        return result

    def _merge_variables(self, runtime_vars: dict[str, str] | None) -> dict[str, str]:
        """Merge runtime variables with config variables.

        Args:
            runtime_vars: Variables passed to run().

        Returns:
            Merged variable dict (runtime overrides config).
        """
        merged = dict(self.config.variables or {})
        if runtime_vars:
            merged.update(runtime_vars)
        return merged

    def _load_prompt_with_variables(
        self,
        variables: dict[str, str],
    ) -> tuple[str, PromptTemplate | None]:
        """Load prompt template and substitute variables.

        Args:
            variables: Variable values for substitution.

        Returns:
            Tuple of (substituted_prompt, template or None).
            Template is None if prompt has no frontmatter.
        """
        input_path = self.config.input_path

        # Try to load as template with frontmatter
        try:
            template = PromptTemplate.from_file(
                input_path,
                default_delimiter=self.settings.get_delimiter_config()
            )

            # Validate variables
            errors = template.validate(variables)
            if errors:
                # Filter to actual errors (not warnings)
                actual_errors = [e for e in errors if not e.startswith("Warning:")]
                if actual_errors:
                    raise ValueError(f"Variable validation failed: {'; '.join(actual_errors)}")

            # Substitute variables
            substituted = template.substitute(variables)
            return substituted, template

        except ValueError as e:
            # If frontmatter parsing fails, treat as plain prompt
            if "Invalid YAML frontmatter" in str(e):
                # Load as plain prompt
                prompt = load_prompt(input_path)
                return prompt, None
            raise

    def _restore_preserved_variables(
        self,
        content: str,
        template: PromptTemplate,
        variables: dict[str, str],
    ) -> str:
        """Restore preserved variable placeholders in optimized output.

        For variables with mode='preserve', replace substituted values
        back with placeholders in the output.

        Args:
            content: Optimized prompt content.
            template: Original prompt template.
            variables: Variables that were substituted.

        Returns:
            Content with preserved variables restored to placeholders.
        """
        preserved_vars = template.get_preserved_variables()
        all_vars = template.get_all_variables()

        result = content
        delimiter = template.delimiter

        for var_name in preserved_vars:
            var_def = all_vars.get(var_name)
            if var_def and var_name in variables:
                # Get the value that was substituted
                value = variables[var_name]
                # Replace the value with the placeholder
                placeholder = f"{delimiter.start}{var_name}{delimiter.end}"
                result = result.replace(value, placeholder)

        return result

    def _setup(self) -> None:
        """Hook: Called before optimization starts."""
        pass

    def _get_provider(self, name: str):
        """Get the configured LM provider."""
        return ProviderFactory.create(name)

    def _get_cache_key(self, prompt: str, optimizer: str, provider: str) -> str:
        """Generate cache key from prompt and configuration."""
        content = f"{provider}:{optimizer}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_optimizer(self, name: str):
        """Get the configured optimizer."""
        return OptimizerFactory.create(name)

    def _optimize(self, prompt: str, optimizer) -> OptimizationResult:
        """Run the actual optimization."""
        import dspy

        trainset = self.config.trainset or []
        # Create default training example if none provided
        if not trainset:
            trainset = [
                dspy.Example(
                    unoptimized_prompt="Write a greeting.",
                    optimized_prompt="Write a greeting.\n\nExample:\nInput: Write a greeting for a friend.\nOutput: Hey there! Great to see you again!",
                ).with_inputs("unoptimized_prompt")
            ]
        return optimizer.optimize(prompt, trainset)

    def _save_result(self, result: OptimizationResult) -> None:
        """Save the optimized prompt."""
        output_path = self.config.output_path
        if output_path.exists():
            version = 1
            while output_path.with_name(f"{output_path.stem}_{version:02d}{output_path.suffix}").exists():
                version += 1
            output_path = output_path.with_name(f"{output_path.stem}_{version:02d}{output_path.suffix}")
        save_prompt(result.optimized_prompt, output_path)

    def _cleanup(self) -> None:
        """Hook: Called after optimization completes."""
        pass