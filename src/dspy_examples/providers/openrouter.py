"""OpenRouter LLM provider adapter."""

import dspy
from dspy_examples.providers.base import LMProvider, LMConfig
from dspy_examples.settings import get_settings


class OpenRouterProvider(LMProvider):
    """Adapter for OpenRouter LLM provider."""

    def __init__(self, config: LMConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "OpenRouterProvider":
        """Create provider from environment variables."""
        settings = get_settings()
        return cls(LMConfig(
            model_name=settings.openrouter_model,
            provider="openrouter",
            api_key=settings.openrouter_api_key,
        ))

    def create_lm(self) -> dspy.LM:
        """Create a DSPy LM instance for OpenRouter."""
        return dspy.LM(
            model=f"openrouter/{self.config.model_name}",
            api_key=self.config.api_key,
        )

    def validate_config(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.config.model_name and self.config.api_key)

    @classmethod
    def provider_name(cls) -> str:
        return "openrouter"