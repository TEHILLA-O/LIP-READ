"""Auto-AVSR visual speech recognition adapter.

Conformer encoder over a 3D-conv ResNet-18 front-end, decoded by a Transformer
decoder with CTC prefix rescoring. Reported word error rate on LRS3 is 20.3%,
so roughly one word in five is wrong even on clean, frontal, studio-quality
video; results on webcam footage are worse. The UI must present output as a
best-effort reading of lip movement.

Reference: Ma et al., "Auto-AVSR: Audio-Visual Speech Recognition with Automatic
Labels", ICASSP 2023.
"""

from __future__ import annotations

import logging
import time

import torch

from ml.config import DecodingConfig, PreprocessingConfig, resolve_device
from ml.decoding.beam_search import JointBeamSearchDecoder
from ml.decoding.text import load_text_transform
from ml.models.auto_avsr.loader import build_e2e, load_state_dict_into
from ml.models.base import (
    Hypothesis,
    ModelNotLoadedError,
    RawModelOutput,
    VisualSpeechModel,
)
from ml.models.checkpoints import DEFAULT_CHECKPOINT, checkpoint_path
from ml.preprocessing.tensor import add_batch_dimension, rois_to_tensor
from ml.types import ModelSpec, MouthTrack

logger = logging.getLogger(__name__)

__all__ = ["AutoAVSRAdapter"]


class AutoAVSRAdapter(VisualSpeechModel):
    """Runs the pretrained Auto-AVSR visual-only model."""

    name = "auto_avsr"

    def __init__(
        self,
        checkpoint_key: str = DEFAULT_CHECKPOINT,
        device: str = "auto",
        preprocessing: PreprocessingConfig | None = None,
        decoding: DecodingConfig | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_key = checkpoint_key
        self.device = torch.device(resolve_device(device))
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.decoding = decoding or DecodingConfig()

        self._model: torch.nn.Module | None = None
        self._decoder: JointBeamSearchDecoder | None = None
        self._text_transform = None
        self._load_report: dict[str, int] = {}

    @property
    def spec(self) -> ModelSpec:
        size = self.preprocessing.network_input_size
        return ModelSpec(
            name=self.name,
            fps=self.preprocessing.target_fps,
            crop_size=(self.preprocessing.crop_width, self.preprocessing.crop_height),
            input_size=(size, size),
            channels=1,
            tensor_layout="(B, T, C, H, W) float32, grayscale, normalised",
            checkpoint_required=True,
            description=(
                "Auto-AVSR Conformer VSR, 20.3% WER on LRS3. English sentences."
            ),
        )

    def load(self) -> None:
        """Build the network and load weights once, at server start."""
        if self._loaded:
            return

        started = time.perf_counter()
        path = checkpoint_path(self.checkpoint_key)
        self._text_transform = load_text_transform()

        model = build_e2e(odim=len(self._text_transform), modality="video")
        self._load_report = load_state_dict_into(model, path)
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        self._decoder = JointBeamSearchDecoder(
            model=model,
            text_transform=self._text_transform,
            config=self.decoding,
            device=self.device,
        )
        self._loaded = True

        logger.info(
            "Loaded Auto-AVSR (%s) on %s in %.1fs: %s",
            self.checkpoint_key,
            self.device,
            time.perf_counter() - started,
            self._load_report,
        )

    def unload(self) -> None:
        self._model = None
        self._decoder = None
        super().unload()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def preprocess(self, track: MouthTrack) -> torch.Tensor:
        """Mouth crops -> ``(1, T, 1, 88, 88)`` normalised grayscale."""
        tensor = rois_to_tensor(track.rois, config=self.preprocessing, device=self.device)
        return add_batch_dimension(tensor)

    @torch.inference_mode()
    def infer(self, tensor: torch.Tensor) -> RawModelOutput:
        """Encode the clip. Decoding is deferred so the debug view can inspect this."""
        if self._model is None:
            raise ModelNotLoadedError("Call load() before infer()")
        if tensor.ndim != 5:
            raise ValueError(f"Expected a (B, T, C, H, W) tensor, got {tensor.shape}")

        tensor = tensor.to(self.device, non_blocking=True)
        features = self._model.frontend(tensor)
        features = self._model.proj_encoder(features)
        encoded, _ = self._model.encoder(features, None)

        ctc_log_probs = None
        if hasattr(self._model.ctc, "log_softmax"):
            ctc_log_probs = self._model.ctc.log_softmax(encoded)

        return RawModelOutput(
            frames_in=int(tensor.shape[1]),
            frames_encoded=int(encoded.shape[1]),
            encoder_features=encoded,
            ctc_log_probs=ctc_log_probs,
            extra={"model": self.name, "checkpoint": self.checkpoint_key},
        )

    def decode(self, raw: RawModelOutput) -> list[Hypothesis]:
        if self._decoder is None:
            raise ModelNotLoadedError("Call load() before decode()")
        if raw.encoder_features is None:
            raise ValueError("Auto-AVSR decoding needs encoder features")
        return self._decoder.decode(raw.encoder_features.squeeze(0))
