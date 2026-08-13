"""Template for a project-specific visual speech model.

Copy this file when plugging in your own network. It documents the contract by
showing the shape of a real implementation, and every method raises rather than
returning placeholder text, so an unfinished adapter can never be mistaken for
a working one.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ml.config import DecodingConfig, PreprocessingConfig, resolve_device
from ml.models.base import Hypothesis, RawModelOutput, VisualSpeechModel
from ml.preprocessing.tensor import add_batch_dimension, rois_to_tensor
from ml.types import ModelSpec, MouthTrack

__all__ = ["CustomVSRAdapter"]


class CustomVSRAdapter(VisualSpeechModel):
    """Skeleton adapter for a custom checkpoint.

    ``preprocess`` is implemented because most visual models want the same
    aligned grayscale mouth crop; override it if yours differs. ``load``,
    ``infer`` and ``decode`` are yours to fill in.
    """

    name = "custom"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        preprocessing: PreprocessingConfig | None = None,
        decoding: DecodingConfig | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(resolve_device(device))
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.decoding = decoding or DecodingConfig()
        self._model: torch.nn.Module | None = None

    @property
    def spec(self) -> ModelSpec:
        size = self.preprocessing.network_input_size
        return ModelSpec(
            name=self.name,
            fps=self.preprocessing.target_fps,
            crop_size=(self.preprocessing.crop_width, self.preprocessing.crop_height),
            input_size=(size, size),
            channels=1,
            tensor_layout="(B, T, C, H, W)",
            checkpoint_required=True,
            description="Custom model slot. Not implemented.",
        )

    def load(self) -> None:
        raise NotImplementedError(
            "CustomVSRAdapter.load is a template. Build your network, load "
            "weights from self.checkpoint_path, move it to self.device, call "
            "eval(), then set self._loaded = True."
        )

    def preprocess(self, track: MouthTrack) -> torch.Tensor:
        """Default: the same normalised grayscale crop Auto-AVSR expects."""
        tensor = rois_to_tensor(
            track.rois, config=self.preprocessing, device=self.device
        )
        return add_batch_dimension(tensor)

    def infer(self, tensor: torch.Tensor) -> RawModelOutput:
        raise NotImplementedError(
            "CustomVSRAdapter.infer is a template. Run your network under "
            "torch.inference_mode() and return a RawModelOutput."
        )

    def decode(self, raw: RawModelOutput) -> list[Hypothesis]:
        raise NotImplementedError(
            "CustomVSRAdapter.decode is a template. Return Hypothesis objects "
            "ordered best-first, with confidence in (0, 1]."
        )
