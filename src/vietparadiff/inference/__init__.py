"""Inference configuration, sampling, and generation APIs."""

from .generator import (
    GenerationOutput,
    SamplingConfig,
    VietParaDiffGenerationConfig,
    generate_paragraph,
    load_generation_config,
)

__all__ = [
    "GenerationOutput",
    "SamplingConfig",
    "VietParaDiffGenerationConfig",
    "generate_paragraph",
    "load_generation_config",
]
