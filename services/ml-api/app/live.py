"""Per-connection state for the live WebSocket path.

Kept out of the route so the socket handler stays a thin transport layer and
this logic — decode, detect, segment, submit — can be tested without a network.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ml.config import AppConfig
from ml.preprocessing.face_detector import FaceTracker
from ml.streaming.buffer import BufferedFrame, RollingFrameBuffer
from ml.streaming.segmenter import (
    LipMotionDetector,
    SegmenterState,
    SegmentEvent,
    UtteranceSegmenter,
)
from ml.types import BoundingBox

logger = logging.getLogger(__name__)

__all__ = ["LiveSession", "FrameObservation"]

#: Extra context taken before an utterance starts, so the first visible movement
#: of the lips is not clipped off the front of the segment.
_LEAD_IN_SECONDS = 0.25


@dataclass
class FrameObservation:
    """What the client is shown about the frame it just sent."""

    mouth_detected: bool
    energy: float
    face_box: BoundingBox | None = None
    mouth_box: BoundingBox | None = None
    #: Inner lip outline in source pixels, when the mesh fitted. For the debug
    #: overlay only; the transcription path does its own landmarking.
    lip_contour: np.ndarray | None = None
    frame_shape: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "mouth_detected": self.mouth_detected,
            "energy": round(self.energy, 4),
            "face_box": self.face_box.to_dict() if self.face_box else None,
            "mouth_box": self.mouth_box.to_dict() if self.mouth_box else None,
            "lip_contour": (
                None
                if self.lip_contour is None
                else np.rint(self.lip_contour).astype(int).tolist()
            ),
            "frame_height": self.frame_shape[0],
            "frame_width": self.frame_shape[1],
        }


@dataclass
class LiveSession:
    """Owns the buffer, tracker and segmenter for one WebSocket connection.

    Every connection gets its own instances because MediaPipe detectors and the
    motion detector are stateful and not thread-safe. Only the shared inference
    engine is common across connections, and that serialises internally.
    """

    config: AppConfig
    buffer: RollingFrameBuffer = field(init=False)
    tracker: FaceTracker = field(init=False)
    motion: LipMotionDetector = field(init=False)
    segmenter: UtteranceSegmenter = field(init=False)

    frames_received: int = 0
    frames_rejected: int = 0
    _started_at: float = field(default_factory=time.monotonic)
    _last_frame_at: float | None = None
    _interval_ema: float | None = None

    def __post_init__(self) -> None:
        preprocessing = self.config.preprocessing
        streaming = self.config.streaming
        self.buffer = RollingFrameBuffer(
            capacity_seconds=streaming.buffer_seconds, fps=preprocessing.target_fps
        )
        self.tracker = FaceTracker()
        self.motion = LipMotionDetector()
        self.segmenter = UtteranceSegmenter(streaming)

    def close(self) -> None:
        self.tracker.close()
        self.motion.close()
        self.buffer.clear()

    # ------------------------------------------------------------------ metrics

    @property
    def observed_fps(self) -> float:
        """Smoothed arrival rate, for the debug view and for resampling checks."""
        if not self._interval_ema:
            return 0.0
        return 1.0 / self._interval_ema

    @property
    def state(self) -> SegmenterState:
        return self.segmenter.state

    def stats(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "frames_rejected": self.frames_rejected,
            "buffer_frames": len(self.buffer),
            "buffer_seconds": round(self.buffer.duration_seconds, 2),
            "observed_fps": round(self.observed_fps, 1),
            "state": self.segmenter.state.value,
            "utterance_seconds": round(self.segmenter.current_duration, 2),
        }

    # -------------------------------------------------------------------- input

    @staticmethod
    def decode_frame(payload: bytes) -> np.ndarray | None:
        """Decode a JPEG/PNG frame from the browser into uint8 RGB.

        Returns None for anything undecodable rather than raising: a single
        corrupt frame on a live stream should be skipped, not close the socket.
        """
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def observe(self, image: np.ndarray, timestamp: float | None = None) -> FrameObservation:
        """Detect the face, measure lip motion and buffer the frame.

        Runs on every frame, but only detection and lip measurement. Alignment
        and the model run later, on whole segments.
        """
        now = time.monotonic() if timestamp is None else timestamp
        if self._last_frame_at is not None:
            interval = max(1e-3, now - self._last_frame_at)
            self._interval_ema = (
                interval
                if self._interval_ema is None
                else 0.9 * self._interval_ema + 0.1 * interval
            )
        self._last_frame_at = now
        self.frames_received += 1

        observation = self.tracker.detect(image, self.frames_received)
        self.buffer.append(image, timestamp=now)

        if observation is None:
            self.motion.reset()
            return FrameObservation(
                mouth_detected=False, energy=0.0, frame_shape=image.shape[:2]
            )

        energy = self.motion.update(image)
        shape = self.motion.last_shape
        # The measured lip contour is a tighter indicator than the keypoint
        # estimate, but the mesh fails on frames the face detector still handles,
        # so the approximation remains the fallback.
        mouth_box = (
            shape.box if shape is not None else _mouth_box(observation.keypoints[3], observation.box)
        )
        return FrameObservation(
            mouth_detected=True,
            energy=energy,
            face_box=observation.box,
            mouth_box=mouth_box,
            lip_contour=None if shape is None else shape.contour,
            frame_shape=image.shape[:2],
        )

    def update_segmenter(self, observation: FrameObservation) -> SegmentEvent | None:
        return self.segmenter.update(observation.energy, self._last_frame_at or 0.0)

    def flush(self) -> SegmentEvent | None:
        return self.segmenter.flush(self._last_frame_at)

    # ------------------------------------------------------------------ segment

    def collect_segment(self, event: SegmentEvent) -> list[np.ndarray]:
        """Pull the frames belonging to a closed utterance, at the model's fps.

        The browser cannot be trusted to deliver exactly 25 fps, and the model
        has no other time base, so frames are chosen by timestamp rather than by
        count. Sending a 30 fps capture straight through would make every
        utterance play back 20% slow to the model.
        """
        window = event.duration + _LEAD_IN_SECONDS
        frames = self.buffer.window(min(window, self.config.streaming.buffer_seconds))
        start = event.start_time - _LEAD_IN_SECONDS
        selected = [frame for frame in frames if frame.timestamp >= start]
        return _resample(selected or frames, self.config.preprocessing.target_fps)


def _mouth_box(mouth_point: np.ndarray, face: BoundingBox) -> BoundingBox:
    """Approximate mouth box for the on-screen indicator.

    Derived from the face box rather than measured, because the live path
    deliberately skips landmark alignment. It marks where the crop will be taken,
    which is all the indicator needs to convey.
    """
    width = max(8, int(round(face.width * 0.45)))
    height = max(8, int(round(face.height * 0.28)))
    return BoundingBox(
        x=int(round(mouth_point[0] - width / 2)),
        y=int(round(mouth_point[1] - height / 2)),
        width=width,
        height=height,
    )


def _resample(frames: list[BufferedFrame], target_fps: int) -> list[np.ndarray]:
    """Nearest-neighbour resample buffered frames onto a fixed grid."""
    if len(frames) < 2:
        return [frame.image for frame in frames]

    period = 1.0 / target_fps
    start, end = frames[0].timestamp, frames[-1].timestamp
    count = int(round((end - start) / period)) + 1
    if count <= 1:
        return [frame.image for frame in frames]

    timestamps = np.fromiter((f.timestamp for f in frames), dtype=np.float64, count=len(frames))
    targets = start + np.arange(count) * period
    indices = np.abs(timestamps[None, :] - targets[:, None]).argmin(axis=1)
    return [frames[int(index)].image for index in indices]
