"""DSPy prompt optimization examples."""

from dspy_examples.settings import Settings, get_settings
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
from dspy_examples.factory.provider_factory import ProviderFactory
from dspy_examples.factory.optimizer_factory import OptimizerFactory
from dspy_examples.template import (
    PromptTemplate,
    VariableDef,
    DelimiterConfig,
    ParsedPrompt,
)

__all__ = [
    "Settings",
    "get_settings",
    "OptimizationPipeline",
    "PipelineConfig",
    "ProviderFactory",
    "OptimizerFactory",
    "PromptTemplate",
    "VariableDef",
    "DelimiterConfig",
    "ParsedPrompt",
]