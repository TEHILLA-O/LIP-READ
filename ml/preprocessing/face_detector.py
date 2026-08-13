"""Face detection and single-speaker tracking.

Auto-AVSR aligns faces on four points — right eye, left eye, nose tip and mouth
centre — which is exactly what MediaPipe's face detector returns, so no separate
landmark model is needed. The tracker keeps the pipeline locked to one speaker
when several faces are visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

import numpy as np

from ml.types import BoundingBox, FaceObservation

__all__ = ["FaceTracker", "KEYPOINT_ORDER"]

#: MediaPipe keypoint indices in the order the alignment step expects them.
KEYPOINT_ORDER: tuple[str, ...] = ("right_eye", "left_eye", "nose_tip", "mouth_centre")

_FULL_RANGE = 1
_SHORT_RANGE = 0


class FaceTracker:
    """Detects one face per frame and keeps it consistent across frames.

    MediaPipe's full-range model handles the typical webcam framing; the
    short-range model is tried as a per-frame fallback because it wins on
    close-up faces that fill the frame. When several faces are present the one
    overlapping the previous track is preferred over the largest, so a bystander
    moving closer to the camera cannot steal the transcript mid-utterance.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_track_iou: float = 0.2,
    ) -> None:
        import mediapipe as mp

        self._solutions = mp.solutions.face_detection
        self._full_range = self._solutions.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=_FULL_RANGE,
        )
        self._short_range = self._solutions.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=_SHORT_RANGE,
        )
        self._min_track_iou = min_track_iou
        self._last_box: BoundingBox | None = None

    def reset(self) -> None:
        """Forget the current track; call between unrelated clips."""
        self._last_box = None

    def close(self) -> None:
        self._full_range.close()
        self._short_range.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def detect(self, image: np.ndarray, frame_index: int = 0) -> FaceObservation | None:
        """Return the tracked face in ``image``, or None if none was found.

        ``image`` must be uint8 RGB; MediaPipe reads the buffer directly.
        """
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 RGB frame, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB frame, got {image.shape}")

        candidates = self._run(self._full_range, image, frame_index)
        if not candidates:
            candidates = self._run(self._short_range, image, frame_index)
        if not candidates:
            return None

        chosen = self._select(candidates)
        self._last_box = chosen.box
        return chosen

    def detect_sequence(self, images: Sequence[np.ndarray]) -> list[FaceObservation | None]:
        """Detect across a whole in-memory clip, resetting the track first."""
        self.reset()
        return [self.detect(image, index) for index, image in enumerate(images)]

    def _run(
        self, detector, image: np.ndarray, frame_index: int
    ) -> list[FaceObservation]:
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        results = detector.process(image)
        if not results.detections:
            return []

        height, width = image.shape[:2]
        observations: list[FaceObservation] = []
        for detection in results.detections:
            location = detection.location_data
            relative = location.relative_bounding_box
            box = BoundingBox(
                x=int(round(relative.xmin * width)),
                y=int(round(relative.ymin * height)),
                width=int(round(relative.width * width)),
                height=int(round(relative.height * height)),
            )
            if box.width <= 0 or box.height <= 0:
                continue
            keypoints = np.array(
                [
                    [
                        location.relative_keypoints[index].x * width,
                        location.relative_keypoints[index].y * height,
                    ]
                    for index in range(len(KEYPOINT_ORDER))
                ],
                dtype=np.float32,
            )
            score = float(detection.score[0]) if detection.score else 0.0
            observations.append(
                FaceObservation(
                    frame_index=frame_index, box=box, keypoints=keypoints, score=score
                )
            )
        return observations

    def _select(self, candidates: list[FaceObservation]) -> FaceObservation:
        if len(candidates) == 1:
            return candidates[0]
        if self._last_box is not None:
            best = max(candidates, key=lambda c: self._last_box.iou(c.box))
            if self._last_box.iou(best.box) >= self._min_track_iou:
                return best
        return max(candidates, key=lambda c: c.box.width * c.box.height)
