"""The model interface the rest of the application programs against.

Nothing outside this package should import a specific research model. Swapping
Auto-AVSR for VALLR or a custom network must be a registry change, not an
application change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ml.types import Hypothesis, ModelSpec, MouthTrack

__all__ = [
    "VisualSpeechModel",
    "Hypothesis",
    "RawModelOutput",
    "CheckpointMissingError",
    "ModelNotLoadedError",
]


class CheckpointMissingError(RuntimeError):
    """Raised when a required pretrained checkpoint is not on disk.

    Carries the download instructions so the failure is actionable rather than
    just a missing-file path.
    """


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before :meth:`VisualSpeechModel.load`."""


@dataclass
class RawModelOutput:
    """Model output before text decoding, kept for the debug view.

    Tensors are optional so adapters that expose different internals (for
    example a phoneme-CTC model) can still populate what they have.
    """

    frames_in: int
    frames_encoded: int
    encoder_features: torch.Tensor | None = None
    ctc_log_probs: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Small, JSON-safe description for the debug panel."""
        info: dict[str, Any] = {
            "frames_in": self.frames_in,
            "frames_encoded": self.frames_encoded,
        }
        if self.encoder_features is not None:
            info["encoder_features_shape"] = list(self.encoder_features.shape)
        if self.ctc_log_probs is not None:
            info["ctc_log_probs_shape"] = list(self.ctc_log_probs.shape)
        return info | self.extra


class VisualSpeechModel(ABC):
    """A loadable visual speech recogniser.

    Lifecycle is ``load()`` once at server start, then ``preprocess`` ->
    ``infer`` -> ``decode`` per utterance. Implementations are expected to be
    usable from a single inference worker thread; they are not required to be
    thread-safe.
    """

    name: ClassVar[str] = "base"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    @abstractmethod
    def spec(self) -> ModelSpec:
        """Frame rate, crop size and tensor layout this model requires."""

    @abstractmethod
    def load(self) -> None:
        """Build the network and load weights. Must be idempotent."""

    @abstractmethod
    def preprocess(self, track: MouthTrack) -> torch.Tensor:
        """Turn a mouth ROI sequence into this model's input tensor."""

    @abstractmethod
    def infer(self, tensor: torch.Tensor) -> RawModelOutput:
        """Run the network. Implementations must disable gradients."""

    @abstractmethod
    def decode(self, raw: RawModelOutput) -> list[Hypothesis]:
        """Turn raw output into ranked transcripts, best first."""

    def transcribe(self, track: MouthTrack) -> tuple[list[Hypothesis], RawModelOutput]:
        """Convenience path through all three stages."""
        self._require_loaded()
        raw = self.infer(self.preprocess(track))
        return self.decode(raw), raw

    def unload(self) -> None:
        """Release weights and GPU memory."""
        self._loaded = False

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError(
                f"{type(self).__name__} must be loaded before inference; call load()."
            )
