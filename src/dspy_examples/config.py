"""DSPy configuration module for Ollama integration."""

import os
from typing import Any

import dspy
from dotenv import load_dotenv


def load_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary with model, base_url, and other settings.
    """
    load_dotenv()

    return {
        "model": os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


def configure_dspy() -> dspy.LM:
    """Configure DSPy to use Ollama language model.

    Returns:
        Configured dspy.LM instance.
    """
    config = load_config()

    lm = dspy.LM(
        model=f"ollama_chat/{config['model']}",
        api_base=config["base_url"],
    )
    dspy.configure(lm=lm)
    return lm