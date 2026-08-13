"""Video to mouth-ROI sequence.

This is the whole Phase 1 chain in one place:

    decode -> detect face -> track -> landmarks -> align -> crop mouth

The result is a :class:`~ml.types.MouthTrack`, which is deliberately independent
of any model: it can be exported to MP4 and inspected before a checkpoint exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType

import numpy as np

from ml.config import PreprocessingConfig
from ml.preprocessing.alignment import (
    MOUTH_KEYPOINT_INDEX,
    cut_patch,
    estimate_alignment,
    transform_points,
    warp_face,
)
from ml.preprocessing.face_detector import FaceTracker
from ml.preprocessing.landmarks import interpolate_missing, smooth_landmarks
from ml.preprocessing.video_io import DecodedFrame, decode_frames, probe
from ml.types import BoundingBox, FaceObservation, MouthTrack

__all__ = ["MouthROIPipeline", "PreprocessingError", "NoFaceDetectedError"]


class PreprocessingError(RuntimeError):
    """Raised when a clip cannot be turned into a mouth sequence."""


class NoFaceDetectedError(PreprocessingError):
    """Raised when no frame in the clip contains a detectable face."""


class MouthROIPipeline:
    """Extracts aligned mouth crops from video files or in-memory frames.

    Holds a MediaPipe detector, so instantiate once and reuse. Not thread-safe:
    give each worker thread its own instance.
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        tracker: FaceTracker | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self._tracker = tracker or FaceTracker()

    def close(self) -> None:
        self._tracker.close()

    def __enter__(self) -> MouthROIPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ public

    def from_video(self, path: str | Path) -> MouthTrack:
        """Extract the mouth sequence from a video file.

        Decodes twice on purpose. Landmark smoothing needs the frames that come
        *after* the one being cropped, and buffering a whole clip of
        full-resolution frames to get them would cost hundreds of megabytes for
        a video the user may have chosen at any resolution.
        """
        metadata = probe(path)
        observations = list(self._detect_pass(path))
        if not observations:
            raise PreprocessingError(f"No frames decoded from {path}")

        landmarks = self._fill_landmarks(observations)
        frames = (frame.image for frame in self._decode(path))
        return self._crop_pass(
            frames=frames,
            landmarks=landmarks,
            boxes=[o.box if o else None for o in observations],
            detected_frames=sum(o is not None for o in observations),
            source_fps=metadata.fps,
        )

    def from_frames(
        self, frames: Sequence[np.ndarray], source_fps: float | None = None
    ) -> MouthTrack:
        """Extract the mouth sequence from frames already in memory.

        Used by the live path, where frames arrive one at a time and the clip is
        a few seconds at most. Frames must already be at the target frame rate.
        """
        if len(frames) < self.config.min_frames:
            raise PreprocessingError(
                f"Need at least {self.config.min_frames} frames, got {len(frames)}"
            )

        observations = self._tracker.detect_sequence(frames)
        landmarks = self._fill_landmarks(observations)
        return self._crop_pass(
            frames=iter(frames),
            landmarks=landmarks,
            boxes=[o.box if o else None for o in observations],
            detected_frames=sum(o is not None for o in observations),
            source_fps=source_fps or float(self.config.target_fps),
        )

    # ----------------------------------------------------------------- internal

    def _decode(self, path: str | Path) -> Iterator[DecodedFrame]:
        return decode_frames(
            path,
            target_fps=self.config.target_fps,
            long_side=self.config.detection_long_side,
            max_frames=self.config.max_frames,
        )

    def _detect_pass(self, path: str | Path) -> Iterator[FaceObservation | None]:
        self._tracker.reset()
        for frame in self._decode(path):
            yield self._tracker.detect(frame.image, frame.index)

    def _fill_landmarks(
        self, observations: Sequence[FaceObservation | None]
    ) -> np.ndarray:
        raw: list[np.ndarray | None] = [
            None if item is None else item.keypoints for item in observations
        ]
        filled = interpolate_missing(raw)
        if filled is None:
            raise NoFaceDetectedError(
                "No face was detected in any frame. Check lighting, framing and "
                "that the speaker is facing the camera."
            )
        return filled

    def _crop_pass(
        self,
        frames: Iterable[np.ndarray],
        landmarks: np.ndarray,
        boxes: list[BoundingBox | None],
        detected_frames: int,
        source_fps: float,
    ) -> MouthTrack:
        config = self.config
        half_height = config.crop_height // 2
        half_width = config.crop_width // 2
        total = landmarks.shape[0]

        rois: list[np.ndarray] = []
        aligned_keypoints = np.empty((total, 4, 2), dtype=np.float32)
        last_transform: np.ndarray | None = None

        for index, image in enumerate(frames):
            if index >= total:
                break
            smoothed = smooth_landmarks(
                landmarks, index, window=config.landmark_smoothing_window
            )
            transform = estimate_alignment(smoothed, target_size=config.aligned_face_size)
            if transform is None:
                if last_transform is None:
                    raise PreprocessingError(
                        f"Could not align the face on frame {index}; the detected "
                        "keypoints are degenerate (near-profile view?)."
                    )
                transform = last_transform
            last_transform = transform

            aligned = warp_face(image, transform, target_size=config.aligned_face_size)
            points = transform_points(smoothed, transform)
            aligned_keypoints[index] = points
            rois.append(
                cut_patch(
                    aligned, points[MOUTH_KEYPOINT_INDEX], half_height, half_width
                )
            )

        if not rois:
            raise PreprocessingError("Cropping produced no frames")

        cropped = len(rois)
        return MouthTrack(
            rois=np.stack(rois, axis=0).astype(np.uint8, copy=False),
            boxes=boxes[:cropped],
            aligned_keypoints=aligned_keypoints[:cropped],
            detected_frames=min(detected_frames, cropped),
            total_frames=cropped,
            source_fps=source_fps,
            target_fps=config.target_fps,
        )
