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


@dataclass
class PipelineConfig:
    """Configuration for the optimization pipeline."""

    provider_name: str | None = None  # None = use settings default
    optimizer_name: str | None = None  # None = use settings default
    input_path: Path = Path("prompts/unoptimized_prompt.md")
    output_path: Path = Path("prompts/optimized_prompt.md")
    use_cache: bool = True
    trainset: list[Any] | None = None  # Training examples


class OptimizationPipeline:
    """Template pattern for prompt optimization workflow."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.settings = get_settings()
        self.cache = OptimizationCache() if self.config.use_cache else None

    def run(self) -> OptimizationResult:
        """Execute the optimization pipeline."""
        # 1. Setup
        self._setup()

        # 2. Load prompt
        prompt = self._load_prompt()

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

        # 7. Cache result
        if self.cache:
            self.cache.set(cache_key, result)

        # 8. Save output
        self._save_result(result)

        # 9. Cleanup
        self._cleanup()

        return result

    def _setup(self) -> None:
        """Hook: Called before optimization starts."""
        pass

    def _load_prompt(self) -> str:
        """Load the input prompt."""
        return load_prompt(self.config.input_path)

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