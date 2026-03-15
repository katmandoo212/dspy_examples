"""DSPy configuration module for backward compatibility."""

import dspy
from dspy_examples.settings import get_settings


def load_config() -> dict[str, str]:
    """Load configuration from environment variables.

    Returns:
        Dictionary with model, base_url, and other settings.

    Note:
        This function is kept for backward compatibility.
        Prefer using get_settings() for new code.
    """
    settings = get_settings()
    return {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
    }


def configure_dspy() -> dspy.LM:
    """Configure DSPy to use the configured language model.

    Returns:
        Configured dspy.LM instance.

    Note:
        This function is kept for backward compatibility.
        Prefer using ProviderFactory.create() for new code.
    """
    from dspy_examples.factory.provider_factory import ProviderFactory

    settings = get_settings()
    provider = ProviderFactory.create(settings.llm_provider)
    lm = provider.create_lm()
    dspy.configure(lm=lm)
    return lm