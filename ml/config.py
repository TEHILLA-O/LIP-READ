"""Configuration for preprocessing, inference and streaming.

Values are plain dataclasses so the preprocessing pipeline can be imported and
tested without pulling in the API service's settings machinery. Every field can
be overridden from the environment via ``VSR_*`` variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

Device = Literal["cuda", "cpu"]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PreprocessingConfig:
    """The input contract of the Auto-AVSR visual front-end.

    These numbers are not free parameters. They reproduce the preprocessing the
    published checkpoints were trained with, so changing one without retraining
    silently degrades accuracy. See ``docs/preprocessing-spec.md``.
    """

    #: Every clip is resampled to this rate; the model has no other time base.
    target_fps: int = 25

    #: Faces are warped into this canonical frame before the mouth is cut out.
    aligned_face_size: int = 256

    #: Mouth patch cut from the aligned face.
    crop_width: int = 96
    crop_height: int = 96

    #: Centre crop applied to the patch immediately before the network.
    network_input_size: int = 88

    #: Grayscale mean/std the checkpoints were normalised with.
    pixel_mean: float = 0.421
    pixel_std: float = 0.165

    #: Landmarks are averaged over +/- (window_margin // 2) frames to stop the
    #: crop jittering, which the front-end would otherwise read as lip motion.
    landmark_smoothing_window: int = 12

    #: Long side the source frames are scaled to before face detection. Caps
    #: memory and detector cost; faces stay large enough to align accurately.
    detection_long_side: int = 640

    #: Hard ceiling on clip length, enforced during decode.
    max_frames: int = 25 * 60

    #: A clip shorter than this cannot contain a usable utterance.
    min_frames: int = 8

    #: Fraction of frames that must yield a face for the clip to be usable.
    min_detection_ratio: float = 0.3

    @classmethod
    def from_env(cls) -> PreprocessingConfig:
        return cls(
            target_fps=_env_int("VSR_TARGET_FPS", 25),
            detection_long_side=_env_int("VSR_DETECTION_LONG_SIDE", 640),
            max_frames=_env_int("VSR_MAX_FRAMES", 25 * 60),
            min_detection_ratio=_env_float("VSR_MIN_DETECTION_RATIO", 0.3),
        )


@dataclass(frozen=True)
class DecodingConfig:
    """Joint CTC/attention beam search settings."""

    beam_size: int = 20
    ctc_weight: float = 0.1
    length_penalty: float = 0.0
    lm_weight: float = 0.0
    #: How many alternative hypotheses to surface alongside the best one.
    n_best: int = 3

    @classmethod
    def from_env(cls) -> DecodingConfig:
        return cls(
            beam_size=_env_int("VSR_BEAM_SIZE", 20),
            ctc_weight=_env_float("VSR_CTC_WEIGHT", 0.1),
            n_best=_env_int("VSR_N_BEST", 3),
        )


@dataclass(frozen=True)
class StreamingConfig:
    """Live-capture buffering and utterance segmentation."""

    #: Rolling buffer capacity in frames.
    buffer_seconds: float = 8.0

    #: Fixed-window fallback bounds for an utterance.
    min_utterance_seconds: float = 1.2
    max_utterance_seconds: float = 4.0

    #: Consecutive quiet seconds that close an utterance.
    silence_hangover_seconds: float = 0.5

    #: Per-frame change in normalised lip aperture above which the mouth counts
    #: as moving. Chosen by ``scripts/tune_motion_thresholds.py``, which sweeps
    #: these against the labelled fixtures; they sit mid-plateau in a range that
    #: segments every speech clip without firing on a still or moving head. The
    #: lower stop threshold holds an utterance open through the brief lip
    #: closures inside normal speech.
    motion_start_threshold: float = 0.016
    motion_stop_threshold: float = 0.008

    #: Depth of the inference queue; beyond this, the oldest job is dropped so
    #: latency stays bounded instead of growing without limit.
    max_queued_jobs: int = 2

    #: Frames older than this when dequeued are discarded rather than inferred.
    max_frame_age_seconds: float = 6.0

    @classmethod
    def from_env(cls) -> StreamingConfig:
        return cls(
            buffer_seconds=_env_float("VSR_BUFFER_SECONDS", 8.0),
            min_utterance_seconds=_env_float("VSR_MIN_UTTERANCE_SECONDS", 1.2),
            max_utterance_seconds=_env_float("VSR_MAX_UTTERANCE_SECONDS", 4.0),
            motion_start_threshold=_env_float("VSR_MOTION_START_THRESHOLD", 0.016),
            motion_stop_threshold=_env_float("VSR_MOTION_STOP_THRESHOLD", 0.008),
            max_queued_jobs=_env_int("VSR_MAX_QUEUED_JOBS", 2),
        )


@dataclass(frozen=True)
class PrivacyConfig:
    """Retention policy. Transient by default; persistence is opt-in."""

    #: When false (the default) uploads and captures are deleted after inference.
    persist_uploads: bool = False
    #: Debug MP4 export is developer-facing and therefore also opt-in.
    allow_debug_export: bool = True
    #: Age at which stray temp artefacts are swept, in case a process died.
    temp_sweep_age_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> PrivacyConfig:
        return cls(
            persist_uploads=_env_bool("VSR_PERSIST_UPLOADS", False),
            allow_debug_export=_env_bool("VSR_ALLOW_DEBUG_EXPORT", True),
        )


@dataclass(frozen=True)
class AppConfig:
    model_name: str = "auto_avsr"
    device: str = "auto"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            model_name=os.environ.get("VSR_MODEL", "auto_avsr"),
            device=os.environ.get("VSR_DEVICE", "auto"),
            preprocessing=PreprocessingConfig.from_env(),
            decoding=DecodingConfig.from_env(),
            streaming=StreamingConfig.from_env(),
            privacy=PrivacyConfig.from_env(),
        )

    def with_device(self, device: str) -> AppConfig:
        return replace(self, device=device)


def resolve_device(requested: str = "auto") -> str:
    """Pick the execution device, degrading to CPU rather than failing."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
