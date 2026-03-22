"""Pydantic Settings configuration for DSPy optimization."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from dspy_examples.template import DelimiterConfig


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Configuration
    llm_provider: Literal["ollama", "openai", "anthropic", "google", "openrouter"] = "ollama"

    # Ollama Configuration
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI Configuration
    openai_api_key: str | None = None
    openai_model: str = "gpt-4"

    # Anthropic Configuration
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-opus-20240229"

    # Google Configuration
    google_api_key: str | None = None
    google_model: str = "gemini-pro"

    # OpenRouter Configuration
    openrouter_api_key: str | None = None
    openrouter_model: str = "anthropic/claude-3-opus"

    # Optimizer Configuration
    optimizer: str = "bootstrap_fewshot"
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 3

    # Variable Substitution Configuration
    variable_delimiter_start: str = "[["
    variable_delimiter_end: str = "]]"

    # Cache Configuration
    use_cache: bool = True
    cache_dir: str = ".cache/optimizations"

    def get_delimiter_config(self) -> DelimiterConfig:
        """Get the delimiter configuration.

        Returns:
            DelimiterConfig with start and end delimiters.
        """
        return DelimiterConfig(
            start=self.variable_delimiter_start,
            end=self.variable_delimiter_end,
        )


# Module-level singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _reset_settings() -> None:
    """Reset the settings singleton (for testing only)."""
    global _settings
    _settings = None