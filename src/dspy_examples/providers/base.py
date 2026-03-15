"""Base interface for LLM providers (Adapter Pattern)."""

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