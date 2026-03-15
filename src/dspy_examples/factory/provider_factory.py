"""Factory for creating LM provider instances."""

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
        """Create a provider instance by name.

        Args:
            provider_name: The provider identifier (e.g., 'ollama', 'openai').

        Returns:
            An LMProvider instance.

        Raises:
            ValueError: If provider_name is not recognized.
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

        provider_class = cls._providers[provider_name]
        return provider_class.from_env()

    @classmethod
    def register(cls, name: str, provider_class: type[LMProvider]) -> None:
        """Register a new provider class.

        Args:
            name: The provider identifier.
            provider_class: The LMProvider subclass to register.
        """
        cls._providers[name] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider identifiers.
        """
        return list(cls._providers.keys())