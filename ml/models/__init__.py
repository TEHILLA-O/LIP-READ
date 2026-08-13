"""Model adapters behind a single interface.

Import :func:`~ml.models.registry.create_model` rather than a concrete adapter,
so that swapping models stays a configuration change.
"""

from ml.models.base import (
    CheckpointMissingError,
    Hypothesis,
    ModelNotLoadedError,
    RawModelOutput,
    VisualSpeechModel,
)
from ml.models.registry import available_models, create_model, register_model

__all__ = [
    "VisualSpeechModel",
    "Hypothesis",
    "RawModelOutput",
    "CheckpointMissingError",
    "ModelNotLoadedError",
    "create_model",
    "available_models",
    "register_model",
]
