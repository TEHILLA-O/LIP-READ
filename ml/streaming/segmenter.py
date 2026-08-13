"""Utterance segmentation for live capture.

Running the model on every frame is both wasteful and wrong: visual speech is
only legible over a sequence. This module decides *when* a sequence is worth
transcribing.

Two layers, deliberately separated so the decision logic can be tested without
video: :class:`LipMotionDetector` turns frames into a motion energy signal, and
:class:`UtteranceSegmenter` turns that signal into start/end events. The
segmenter also works as a plain fixed-window chunker when no energy is supplied.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import TracebackType

import numpy as np

from ml.config import StreamingConfig
from ml.preprocessing.mouth_landmarks import MouthLandmarker, MouthShape

__all__ = [
    "LipMotionDetector",
    "UtteranceSegmenter",
    "SegmentEvent",
    "SegmentReason",
    "SegmenterState",
]


class SegmenterState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


class SegmentReason(str, Enum):
    #: Lip motion fell below the stop threshold for the hangover period.
    SILENCE = "silence"
    #: The utterance hit the maximum window length and was cut.
    MAX_LENGTH = "max_length"
    #: The stream ended while an utterance was open.
    STREAM_END = "stream_end"


@dataclass(frozen=True)
class SegmentEvent:
    """A closed utterance, ready to transcribe."""

    start_time: float
    end_time: float
    reason: SegmentReason

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class LipMotionDetector:
    """Turns lip aperture over time into a motion energy.

    Energy is how fast the lips are opening or closing, averaged over the last
    few frames. Because aperture is normalised by inter-ocular distance, the
    signal ignores head translation, distance from the camera and head tilt, and
    responds only to the mouth changing shape.

    Measured on the project's fixtures at 25 fps, per-frame energy is around
    0.0003 for a still face, 0.004-0.013 while the head moves with the mouth
    closed, and 0.008-0.08 during speech. Head motion sitting an order of
    magnitude below speech is the whole point of the geometric signal: a frame
    difference around the mouth ranks it *above* speech.

    Energy is per frame rather than per second, so it scales with capture rate.
    The browser pump targets the model's 25 fps; a much slower capture reads as
    proportionally more motion.
    """

    def __init__(
        self,
        landmarker: MouthLandmarker | None = None,
        smoothing_frames: int = 3,
    ) -> None:
        self._landmarker = landmarker if landmarker is not None else MouthLandmarker()
        self._owns_landmarker = landmarker is None
        self._recent: deque[float] = deque(maxlen=max(1, smoothing_frames))
        self._previous_aperture: float | None = None
        self.last_shape: MouthShape | None = None

    def reset(self) -> None:
        self._previous_aperture = None
        self.last_shape = None
        self._recent.clear()

    def close(self) -> None:
        if self._owns_landmarker:
            self._landmarker.close()

    def __enter__(self) -> LipMotionDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def update(self, image: np.ndarray) -> float:
        """Return this frame's motion energy, and record the measured shape.

        A frame the mesh cannot fit resets the detector and scores zero, so a
        speaker leaving the frame reads as silence rather than as motion, and the
        first frame back does not register the gap as a jump.
        """
        shape = self._landmarker.measure(image)
        self.last_shape = shape
        if shape is None:
            self.reset()
            return 0.0

        previous = self._previous_aperture
        self._previous_aperture = shape.aperture
        if previous is None:
            return 0.0

        self._recent.append(abs(shape.aperture - previous))
        return float(np.mean(self._recent))


class UtteranceSegmenter:
    """Decides when an utterance starts and ends.

    Hysteresis (a higher threshold to start than to stop) plus a hangover period
    stops the segmenter flickering during the brief closures inside normal
    speech. Utterances are always clamped to
    ``[min_utterance_seconds, max_utterance_seconds]``.
    """

    def __init__(self, config: StreamingConfig | None = None) -> None:
        self.config = config or StreamingConfig()
        self.state = SegmenterState.IDLE
        self._start_time: float | None = None
        self._last_active_time: float | None = None
        self._last_time: float = 0.0

    def reset(self) -> None:
        self.state = SegmenterState.IDLE
        self._start_time = None
        self._last_active_time = None

    @property
    def current_duration(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, self._last_time - self._start_time)

    def update(self, energy: float, timestamp: float) -> SegmentEvent | None:
        """Feed one frame's motion energy; returns an event when one closes."""
        config = self.config
        self._last_time = timestamp

        if self.state is SegmenterState.IDLE:
            if energy >= config.motion_start_threshold:
                self.state = SegmenterState.SPEAKING
                self._start_time = timestamp
                self._last_active_time = timestamp
            return None

        # SPEAKING
        assert self._start_time is not None
        if energy >= config.motion_stop_threshold:
            self._last_active_time = timestamp

        duration = timestamp - self._start_time
        if duration >= config.max_utterance_seconds:
            return self._close(timestamp, SegmentReason.MAX_LENGTH)

        quiet_for = timestamp - (self._last_active_time or timestamp)
        if (
            quiet_for >= config.silence_hangover_seconds
            and duration >= config.min_utterance_seconds
        ):
            return self._close(timestamp, SegmentReason.SILENCE)

        # Too short to be speech and already quiet: abandon without emitting.
        if quiet_for >= config.silence_hangover_seconds:
            self.reset()
        return None

    def flush(self, timestamp: float | None = None) -> SegmentEvent | None:
        """Close an open utterance because the stream ended."""
        if self.state is not SegmenterState.SPEAKING or self._start_time is None:
            return None
        end = self._last_time if timestamp is None else timestamp
        if end - self._start_time < self.config.min_utterance_seconds:
            self.reset()
            return None
        return self._close(end, SegmentReason.STREAM_END)

    def _close(self, end_time: float, reason: SegmentReason) -> SegmentEvent:
        assert self._start_time is not None
        event = SegmentEvent(
            start_time=self._start_time, end_time=end_time, reason=reason
        )
        self.reset()
        return event
