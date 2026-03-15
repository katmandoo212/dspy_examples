# DSPy Optimization Architecture Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create implementation plan after this design is approved.

**Goal:** Refactor DSPy optimization code to use Gang of Four design patterns for extensibility, enabling easy addition of new optimizers and LLM providers.

**Architecture:** Strategy pattern for optimizers, Adapter pattern for LLM providers, Factory pattern for creation, Template pattern for processing pipeline, Memento pattern for caching results.

**Tech Stack:** Python 3.13+, DSPy, Pydantic Settings, python-dotenv

---

## Design Patterns Summary

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Strategy** | `optimizers/` | Swap optimization algorithms |
| **Adapter** | `providers/` | Multiple LLM backends (Ollama, OpenAI, Anthropic, Google, OpenRouter) |
| **Factory** | `factory/` | Create providers & optimizers |
| **Template** | `pipeline.py` | Standard optimization workflow |
| **Memento** | `cache.py` | Save/restore results |
| **Command** | `commands/` | (Future) Batch processing |
| **Observer** | `observers/` | (Future) Progress tracking |
| **Builder** | `builders/` | (Future) Complex configurations |

---

## Project Structure

```
src/dspy_examples/
├── providers/                    # Adapter Pattern
│   ├── __init__.py
│   ├── base.py                   # LMProvider interface
│   ├── ollama.py                 # Ollama adapter
│   ├── openai.py                 # OpenAI adapter
│   ├── anthropic.py              # Anthropic adapter
│   ├── google.py                 # Google/Gemini adapter
│   └── openrouter.py             # OpenRouter adapter
│
├── optimizers/                   # Strategy Pattern
│   ├── __init__.py
│   ├── base.py                   # PromptOptimizer interface
│   ├── bootstrap_fewshot.py
│   └── bootstrap_random.py       # BootstrapFewShotWithRandomSearch
│
├── factory/                      # Factory Pattern
│   ├── __init__.py
│   ├── provider_factory.py        # Create LMProvider instances
│   └── optimizer_factory.py       # Create optimizer instances
│
├── pipeline.py                   # Template Pattern
├── cache.py                      # Memento Pattern
├── commands/                     # (Future) Command Pattern
│   └── __init__.py
├── observers/                    # (Future) Observer Pattern
│   └── __init__.py
├── builders/                      # (Future) Builder Pattern
│   └── __init__.py
│
├── config.py                     # Pydantic Settings + python-dotenv
├── prompts.py                    # (existing)
└── main.py                        # Entry point
```

---

## Core Interfaces

### LMProvider Interface (Adapter Pattern)

```python
# providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import dspy

@dataclass
class LMConfig:
    """Configuration for a language model."""
    model_name: str
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

class LMProvider(ABC):
    """Adapter interface for LLM providers."""

    @abstractmethod
    def create_lm(self) -> dspy.LM:
        """Create a DSPy LM instance for this provider."""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Check if configuration is valid (API keys present, etc.)."""
        ...

    @classmethod
    @abstractmethod
    def from_env(cls) -> "LMProvider":
        """Create provider from environment variables."""
        ...

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str:
        """Return the provider identifier (e.g., 'ollama', 'openai')."""
        ...
```

### PromptOptimizer Interface (Strategy Pattern)

```python
# optimizers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import dspy

@dataclass
class OptimizationResult:
    """Result of a prompt optimization."""
    optimized_prompt: str
    optimizer_name: str
    provider_name: str
    original_length: int
    optimized_length: int
    optimization_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None

@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    name: str
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 3
    num_threads: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

class PromptOptimizer(ABC):
    """Strategy interface for prompt optimizers."""

    @abstractmethod
    def optimize(self, prompt: str, trainset: list[dspy.Example]) -> OptimizationResult:
        """Optimize a prompt using this technique."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return optimizer identifier (e.g., 'bootstrap_fewshot')."""
        ...

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description."""
        ...

    @abstractmethod
    def get_config(self) -> OptimizerConfig:
        """Return current configuration."""
        ...
```

---

## Template Pattern - Optimization Pipeline

```python
# pipeline.py
from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
from typing import Any

from dspy_examples.cache import OptimizationCache
from dspy_examples.factory.provider_factory import ProviderFactory
from dspy_examples.factory.optimizer_factory import OptimizerFactory
from dspy_examples.optimizers.base import OptimizationResult, OptimizerConfig
from dspy_examples.prompts import load_prompt, save_prompt
from dspy_examples.config import get_settings

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

    def __init__(self, config: PipelineConfig | None = None):
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
        trainset = self.config.trainset or []
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
```

---

## Memento Pattern - Optimization Cache

```python
# cache.py
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Any

from dspy_examples.optimizers.base import OptimizationResult

class OptimizationCache:
    """Memento pattern for caching optimization results."""

    def __init__(self, cache_dir: Path | None = None):
        from dspy_examples.config import get_settings
        settings = get_settings()
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> OptimizationResult | None:
        """Retrieve cached result if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            return OptimizationResult(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, cache_key: str, result: OptimizationResult) -> None:
        """Save optimization result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        data = asdict(result)
        data["cached_at"] = datetime.now().isoformat()

        cache_file.write_text(json.dumps(data, indent=2, default=str))

    def list_all(self) -> list[dict[str, Any]]:
        """List all cached optimizations with metadata."""
        results = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                results.append({
                    "cache_key": cache_file.stem,
                    "optimizer_name": data.get("optimizer_name"),
                    "provider_name": data.get("provider_name"),
                    "cached_at": data.get("cached_at"),
                    "original_length": data.get("original_length"),
                    "optimized_length": data.get("optimized_length"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def compare(self, *cache_keys: str) -> list[OptimizationResult]:
        """Compare multiple cached optimizations."""
        results = []
        for key in cache_keys:
            if result := self.get(key):
                results.append(result)
        return results
```

---

## Factory Pattern - Provider & Optimizer Factories

```python
# factory/provider_factory.py
from dspy_examples.providers.base import LMProvider
from dspy_examples.providers.ollama import OllamaProvider
from dspy_examples.providers.openai import OpenAIProvider
from dspy_examples.providers.anthropic import AnthropicProvider
from dspy_examples.providers.google import GoogleProvider
from dspy_examples.providers.openrouter import OpenRouterProvider

class ProviderFactory:
    """Factory for creating LM provider instances."""

    _providers: dict[str, type[LMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create(cls, provider_name: str) -> LMProvider:
        """Create a provider instance by name."""
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

        provider_class = cls._providers[provider_name]
        return provider_class.from_env()

    @classmethod
    def register(cls, name: str, provider_class: type[LMProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())


# factory/optimizer_factory.py
from dspy_examples.optimizers.base import PromptOptimizer, OptimizerConfig
from dspy_examples.optimizers.bootstrap_fewshot import BootstrapFewShotOptimizer
from dspy_examples.optimizers.bootstrap_random import BootstrapRandomOptimizer

class OptimizerFactory:
    """Factory for creating optimizer instances."""

    _optimizers: dict[str, type[PromptOptimizer]] = {
        "bootstrap_fewshot": BootstrapFewShotOptimizer,
        "bootstrap_random": BootstrapRandomOptimizer,
    }

    @classmethod
    def create(cls, optimizer_name: str, config: OptimizerConfig | None = None) -> PromptOptimizer:
        """Create an optimizer instance by name."""
        if optimizer_name not in cls._optimizers:
            available = ", ".join(cls._optimizers.keys())
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: {available}")

        optimizer_class = cls._optimizers[optimizer_name]

        if config:
            return optimizer_class(config)
        return optimizer_class()

    @classmethod
    def register(cls, name: str, optimizer_class: type[PromptOptimizer]) -> None:
        """Register a new optimizer class."""
        cls._optimizers[name] = optimizer_class

    @classmethod
    def list_optimizers(cls) -> list[str]:
        """List all registered optimizer names."""
        return list(cls._optimizers.keys())

    @classmethod
    def auto_select(cls, prompt: str) -> PromptOptimizer:
        """Auto-select best optimizer based on prompt characteristics."""
        # Simple heuristic: use bootstrap_fewshot for prompts < 1000 chars
        if len(prompt) < 1000:
            return cls.create("bootstrap_fewshot")
        return cls.create("bootstrap_random")
```

---

## Configuration with Pydantic Settings

```python
# config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

# Load .env file before reading environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider Configuration
    llm_provider: Literal["ollama", "openai", "anthropic", "google", "openrouter"] = "ollama"

    # Ollama Configuration
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI Configuration
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4"

    # Anthropic Configuration
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-opus-20240229"

    # Google Configuration
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = "gemini-pro"

    # OpenRouter Configuration
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = "anthropic/claude-3-opus"

    # Optimizer Configuration
    optimizer: str = "bootstrap_fewshot"
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 3

    # Cache Configuration
    use_cache: bool = True
    cache_dir: str = ".cache/optimizations"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Module-level singleton
_settings: Settings | None = None

def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        load_dotenv()
        _settings = Settings()
    return _settings
```

---

## Example Provider Implementation

```python
# providers/ollama.py
import dspy
from dspy_examples.providers.base import LMProvider, LMConfig
from dspy_examples.config import get_settings

class OllamaProvider(LMProvider):
    """Adapter for Ollama LLM provider."""

    def __init__(self, config: LMConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "OllamaProvider":
        """Create provider from environment variables."""
        settings = get_settings()
        return cls(LMConfig(
            model_name=settings.ollama_model,
            provider="ollama",
            base_url=settings.ollama_base_url,
        ))

    def create_lm(self) -> dspy.LM:
        """Create a DSPy LM instance for Ollama."""
        return dspy.LM(
            model=f"ollama_chat/{self.config.model_name}",
            api_base=self.config.base_url,
        )

    def validate_config(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.config.model_name and self.config.base_url)

    @classmethod
    def provider_name(cls) -> str:
        return "ollama"
```

---

## Usage Examples

### Basic Usage

```python
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

# Auto-select optimizer, use default provider from .env
pipeline = OptimizationPipeline()
result = pipeline.run()
print(f"Optimized with {result.optimizer_name}")
```

### Explicit Configuration

```python
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

pipeline = OptimizationPipeline(PipelineConfig(
    provider_name="openai",
    optimizer_name="bootstrap_random",
    input_path="prompts/unoptimized_prompt.md",
    output_path="prompts/optimized_prompt.md",
))
result = pipeline.run()
```

### Environment Variables

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
OPTIMIZER=bootstrap_fewshot
```

---

## Future Patterns

### Command Pattern (Batch Processing)
- Location: `commands/`
- Purpose: Encapsulate optimization requests for batch processing multiple prompts

### Observer Pattern (Progress Tracking)
- Location: `observers/`
- Purpose: Progress callbacks, logging, metrics during optimization

### Builder Pattern (Complex Configurations)
- Location: `builders/`
- Purpose: Fluent API for complex optimizer configurations