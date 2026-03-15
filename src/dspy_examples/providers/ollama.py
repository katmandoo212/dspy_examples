"""Ollama LLM provider adapter."""

import dspy
from dspy_examples.providers.base import LMProvider, LMConfig
from dspy_examples.settings import get_settings


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