"""VALLR adapter — interface only, not implemented.

VALLR (Thomas et al., ICCV 2025) is two-stage: a video transformer with a CTC
head predicts phonemes, then a fine-tuned LLM reconstructs sentences. It reports
18.7% WER on LRS3, better than Auto-AVSR, and its phoneme intermediate would
give the debug view something far more interpretable than sentence tokens.

It is not wired up because it needs two things this prototype does not ship:
the released weights (CC BY-NC 4.0, research use only) and an LLM for the
phoneme-to-sentence stage. The class exists so that adding it later is a
registry entry rather than a change to the application.

To implement:
  1. Confirm the mouth ROI geometry VALLR's ``face_cropper.py`` produces; if it
     differs from Auto-AVSR's 96x96 aligned crop, add its own ROI settings
     rather than reusing :class:`~ml.config.PreprocessingConfig` defaults.
  2. Load the video-to-phoneme network and CTC-decode to a phoneme sequence.
  3. Run the phoneme-to-sentence LLM and map its output plus the CTC posterior
     into :class:`~ml.models.base.Hypothesis` objects.
"""

from __future__ import annotations

import torch

from ml.models.base import Hypothesis, RawModelOutput, VisualSpeechModel
from ml.types import ModelSpec, MouthTrack

__all__ = ["VALLRAdapter"]

_NOT_IMPLEMENTED = (
    "VALLRAdapter is an interface stub. Implementing it requires the VALLR "
    "checkpoint (https://github.com/MarshallT-99/VALLR, CC BY-NC 4.0) and the "
    "phoneme-to-sentence LLM. Use model 'auto_avsr' instead."
)


class VALLRAdapter(VisualSpeechModel):
    """Planned phoneme-centric adapter. Every method raises."""

    name = "vallr"

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            fps=25,
            crop_size=(96, 96),
            input_size=(88, 88),
            channels=1,
            tensor_layout="(B, T, C, H, W) — to be confirmed against the release",
            checkpoint_required=True,
            description="VALLR video-to-phoneme + LLM. Not implemented.",
        )

    def load(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def preprocess(self, track: MouthTrack) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def infer(self, tensor: torch.Tensor) -> RawModelOutput:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def decode(self, raw: RawModelOutput) -> list[Hypothesis]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
