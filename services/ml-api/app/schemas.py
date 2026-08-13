"""Response models.

The transcription payload is fixed by the project's result contract; it always
carries confidence and ``mouth_detected`` so no caller can mistake a guess for a
certainty.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AlternativeModel(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptionResponse(BaseModel):
    """Structured inference result."""

    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[AlternativeModel] = Field(default_factory=list)
    processing_ms: int = 0
    frames_processed: int = 0
    mouth_detected: bool = False

    #: Present when no transcript could be produced, with the reason.
    error: str | None = None
    #: Timings and preprocessing detail for the debug view.
    diagnostics: dict[str, Any] | None = None
    #: Debug artefact identifiers, when a debug export was requested.
    debug: dict[str, str] | None = None


class ModelSpecModel(BaseModel):
    name: str
    fps: int
    crop_size: list[int]
    input_size: list[int]
    channels: int
    tensor_layout: str
    checkpoint_required: bool
    description: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    #: Why the model is unavailable, when it is.
    error: str | None = None


class ModelsResponse(BaseModel):
    active: str
    spec: ModelSpecModel | None = None
    #: Registered adapters mapped to whether they are implemented.
    available: dict[str, bool]


class PreprocessResponse(BaseModel):
    """Preprocessing-only result, for verifying the pipeline without a model."""

    frames_processed: int
    detected_frames: int
    detection_ratio: float
    mouth_detected: bool
    duration_seconds: float
    source_fps: float
    target_fps: int
    roi_shape: list[int]
    processing_ms: int
    debug: dict[str, str] | None = None
    error: str | None = None
