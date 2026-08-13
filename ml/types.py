"""Shared data types passed between preprocessing, models and the API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in source-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def iou(self, other: BoundingBox) -> float:
        ax2, ay2 = self.x + self.width, self.y + self.height
        bx2, by2 = other.x + other.width, other.y + other.height
        inter_w = max(0, min(ax2, bx2) - max(self.x, other.x))
        inter_h = max(0, min(ay2, by2) - max(self.y, other.y))
        intersection = inter_w * inter_h
        if intersection == 0:
            return 0.0
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union else 0.0

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class FaceObservation:
    """One frame's face detection.

    ``keypoints`` holds the four points Auto-AVSR aligns on, in this exact
    order: right eye, left eye, nose tip, mouth centre.
    """

    frame_index: int
    box: BoundingBox
    keypoints: np.ndarray  # float32, shape (4, 2)
    score: float


@dataclass
class MouthTrack:
    """The mouth region sequence extracted from a clip.

    ``rois`` is the model-facing product; the rest exists so the debug view can
    explain what the model actually saw.
    """

    #: uint8 RGB mouth crops, shape (T, crop_h, crop_w, 3).
    rois: np.ndarray
    #: Per-frame face boxes in source coordinates; None where interpolated.
    boxes: list[BoundingBox | None]
    #: Per-frame aligned landmarks in the 256x256 canonical face.
    aligned_keypoints: np.ndarray
    #: Frames where a face was genuinely detected rather than filled in.
    detected_frames: int
    total_frames: int
    source_fps: float
    target_fps: int

    @property
    def detection_ratio(self) -> float:
        return self.detected_frames / self.total_frames if self.total_frames else 0.0

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.target_fps if self.target_fps else 0.0


@dataclass
class Hypothesis:
    """One candidate transcript produced by a decoder.

    Lives here rather than with the model adapters because both the decoding and
    the model layers need it, and a shared data type is the wrong thing to make
    either of them depend on the other for.

    ``score`` is the raw beam score (log domain, so negative). ``confidence`` is
    that score length-normalised into (0, 1] for display: it ranks hypotheses
    sensibly but is not a calibrated probability of being correct.
    """

    text: str
    score: float
    confidence: float
    token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Alternative:
    """A non-best beam-search hypothesis."""

    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": round(self.confidence, 4)}


@dataclass
class TranscriptionResult:
    """Structured inference output.

    ``confidence`` is a length-normalised beam score mapped to (0, 1]. It ranks
    hypotheses and flags weak ones; it is not a calibrated probability that the
    transcript is correct. Callers must present it as model confidence, never as
    certainty about what was said.
    """

    text: str = ""
    confidence: float = 0.0
    alternatives: list[Alternative] = field(default_factory=list)
    processing_ms: int = 0
    frames_processed: int = 0
    mouth_detected: bool = False

    #: Populated when inference could not run (no face, clip too short, ...).
    error: str | None = None
    #: Free-form diagnostics for the debug view; never required by the UI.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "processing_ms": self.processing_ms,
            "frames_processed": self.frames_processed,
            "mouth_detected": self.mouth_detected,
        }
        if self.error:
            payload["error"] = self.error
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload

    @classmethod
    def failure(cls, error: str, **kwargs: Any) -> TranscriptionResult:
        """Build an explicit no-transcript result.

        Used wherever the pipeline cannot produce speech, so that callers get a
        stated reason instead of an empty string that looks like silence.
        """
        return cls(text="", confidence=0.0, error=error, **kwargs)


@dataclass(frozen=True)
class ModelSpec:
    """What a model adapter needs from preprocessing, and what it reports back."""

    name: str
    fps: int
    crop_size: tuple[int, int]
    input_size: tuple[int, int]
    channels: int
    #: Human-readable tensor layout, e.g. "(B, T, C, H, W)".
    tensor_layout: str
    checkpoint_required: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "crop_size": list(self.crop_size),
            "input_size": list(self.input_size),
        }


def sequence_to_uint8(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Stack frames into one contiguous uint8 array without per-frame copies."""
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
