"""Inference orchestration.

Owns the one place where preprocessing, a model adapter and decoding meet, and
where every outcome — including failure — becomes a
:class:`~ml.types.TranscriptionResult`. Callers never see an exception for the
ordinary cases (no face, clip too short); they see a result with ``error`` set,
so the UI can explain itself instead of showing a blank transcript.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ml.config import AppConfig
from ml.models.base import CheckpointMissingError, VisualSpeechModel
from ml.models.registry import create_model
from ml.preprocessing.pipeline import MouthROIPipeline, PreprocessingError
from ml.types import Alternative, MouthTrack, TranscriptionResult

logger = logging.getLogger(__name__)

__all__ = ["InferenceEngine"]


class InferenceEngine:
    """Loads a model once and transcribes clips.

    Work is serialised behind a single lock. That is required, not just
    convenient: the MediaPipe detector inside the pipeline is not thread-safe,
    and one GPU copy of a 250M-parameter model cannot serve concurrent requests
    anyway. Callers that need concurrency should queue (see
    :mod:`ml.streaming.worker`) rather than share this object unsynchronised.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        model: VisualSpeechModel | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self._model = model or create_model(
            self.config.model_name,
            device=self.config.device,
            preprocessing=self.config.preprocessing,
            decoding=self.config.decoding,
        )
        self._pipeline = MouthROIPipeline(self.config.preprocessing)
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # ---------------------------------------------------------------- lifecycle

    def load(self) -> None:
        """Load weights at start-up so no request pays the cost.

        A missing checkpoint is recorded rather than raised: the service should
        still start and serve health and preprocessing endpoints, and say
        clearly why transcription is unavailable.
        """
        try:
            self._model.load()
            self._load_error = None
        except CheckpointMissingError as exc:
            self._load_error = str(exc)
            logger.error("Model unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001 - startup must not crash the service
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Unexpected error while loading the model")

    def close(self) -> None:
        self._pipeline.close()
        self._model.unload()

    @property
    def is_ready(self) -> bool:
        return self._model.is_loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model(self) -> VisualSpeechModel:
        return self._model

    def status(self) -> dict[str, object]:
        return {
            "model": self._model.name,
            "ready": self.is_ready,
            "spec": self._model.spec.to_dict(),
            "error": self._load_error,
        }

    # ----------------------------------------------------------------- inference

    def transcribe_video(
        self, path: str | Path, want_debug: bool = False
    ) -> tuple[TranscriptionResult, MouthTrack | None]:
        """Transcribe a video file. Returns the result and, optionally, the track."""
        started = time.perf_counter()
        with self._lock:
            try:
                track = self._pipeline.from_video(path)
            except PreprocessingError as exc:
                return (
                    TranscriptionResult.failure(
                        str(exc),
                        processing_ms=_elapsed_ms(started),
                        mouth_detected=False,
                    ),
                    None,
                )
            result = self._transcribe_track(track, started)
        return result, (track if want_debug else None)

    def transcribe_frames(
        self, frames: Sequence[np.ndarray]
    ) -> tuple[TranscriptionResult, MouthTrack | None]:
        """Transcribe frames already decoded at the model's frame rate."""
        started = time.perf_counter()
        with self._lock:
            try:
                track = self._pipeline.from_frames(frames)
            except PreprocessingError as exc:
                return (
                    TranscriptionResult.failure(
                        str(exc),
                        processing_ms=_elapsed_ms(started),
                        mouth_detected=False,
                    ),
                    None,
                )
            return self._transcribe_track(track, started), track

    def _transcribe_track(
        self, track: MouthTrack, started: float
    ) -> TranscriptionResult:
        minimum = self.config.preprocessing.min_frames
        mouth_detected = (
            track.detection_ratio >= self.config.preprocessing.min_detection_ratio
        )

        if not mouth_detected:
            return TranscriptionResult.failure(
                "The mouth was only visible in "
                f"{track.detection_ratio:.0%} of frames, which is too little to "
                "read. Face the camera and keep your mouth in view.",
                processing_ms=_elapsed_ms(started),
                frames_processed=track.total_frames,
                mouth_detected=False,
            )
        if track.total_frames < minimum:
            return TranscriptionResult.failure(
                f"Clip is {track.total_frames} frames; at least {minimum} are needed.",
                processing_ms=_elapsed_ms(started),
                frames_processed=track.total_frames,
                mouth_detected=True,
            )
        if not self.is_ready:
            return TranscriptionResult.failure(
                self._load_error or "The speech model is not loaded.",
                processing_ms=_elapsed_ms(started),
                frames_processed=track.total_frames,
                mouth_detected=True,
            )

        inference_started = time.perf_counter()
        hypotheses, raw = self._model.transcribe(track)
        inference_ms = _elapsed_ms(inference_started)

        if not hypotheses:
            return TranscriptionResult.failure(
                "The model produced no transcript for this clip.",
                processing_ms=_elapsed_ms(started),
                frames_processed=track.total_frames,
                mouth_detected=True,
            )

        best, *rest = hypotheses
        return TranscriptionResult(
            text=best.text,
            confidence=best.confidence,
            alternatives=[Alternative(h.text, h.confidence) for h in rest],
            processing_ms=_elapsed_ms(started),
            frames_processed=track.total_frames,
            mouth_detected=True,
            diagnostics={
                "inference_ms": inference_ms,
                "detection_ratio": round(track.detection_ratio, 3),
                "duration_seconds": round(track.duration_seconds, 2),
                "source_fps": round(track.source_fps, 2),
                "target_fps": track.target_fps,
                "beam_score": round(best.score, 3),
                "token_count": len(best.token_ids),
                "raw_output": raw.summary(),
            },
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
