"""AV-HuBERT adapter — interface only, not implemented.

AV-HuBERT (Shi et al., 2022) is the research baseline much of this field builds
on, and its preprocessing is the ancestor of the pipeline in
:mod:`ml.preprocessing`. It is deliberately not a runtime dependency: the
released code is pinned to an old Fairseq fork whose environment conflicts with
the PyTorch version used here.

If it is ever needed, run it out of process behind this interface — a separate
environment exposing an HTTP or gRPC endpoint that this adapter calls — rather
than forcing the whole service back onto Fairseq.
"""

from __future__ import annotations

import torch

from ml.models.base import Hypothesis, RawModelOutput, VisualSpeechModel
from ml.types import ModelSpec, MouthTrack

__all__ = ["AVHubertAdapter"]

_NOT_IMPLEMENTED = (
    "AVHubertAdapter is an interface stub. AV-HuBERT requires a legacy Fairseq "
    "environment that is intentionally not installed here; run it out of process "
    "if needed. Use model 'auto_avsr' instead."
)


class AVHubertAdapter(VisualSpeechModel):
    """Planned AV-HuBERT adapter. Every method raises."""

    name = "av_hubert"

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            fps=25,
            crop_size=(96, 96),
            input_size=(88, 88),
            channels=1,
            tensor_layout="(B, T, C, H, W)",
            checkpoint_required=True,
            description="AV-HuBERT, research reference only. Not implemented.",
        )

    def load(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def preprocess(self, track: MouthTrack) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def infer(self, tensor: torch.Tensor) -> RawModelOutput:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def decode(self, raw: RawModelOutput) -> list[Hypothesis]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
