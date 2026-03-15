"""DSPy prompt optimization examples."""

from dspy_examples.settings import Settings, get_settings
from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig
from dspy_examples.factory.provider_factory import ProviderFactory
from dspy_examples.factory.optimizer_factory import OptimizerFactory

__all__ = [
    "Settings",
    "get_settings",
    "OptimizationPipeline",
    "PipelineConfig",
    "ProviderFactory",
    "OptimizerFactory",
]