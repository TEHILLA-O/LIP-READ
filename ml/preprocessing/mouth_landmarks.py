"""Lip geometry from MediaPipe FaceMesh, for the live path only.

The offline pipeline aligns faces from the six-point face detector and never
needs a dense mesh. Segmentation does, because deciding *when* someone is
speaking from frame differences does not work: moving your head changes far more
pixels around the mouth than talking does, so an appearance signal ranks head
motion above speech. Lip aperture measured between the inner lip landmarks and
divided by the inter-ocular distance is invariant to where the head is, how far
away it is and how it is tilted, which is exactly the invariance the decision
needs.

This module is deliberately not wired into :mod:`ml.preprocessing.pipeline`; the
transcription path keeps using the detector it was validated with.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import numpy as np

from ml.types import BoundingBox

__all__ = ["MouthLandmarker", "MouthShape", "INNER_LIP_CONTOUR"]

# FaceMesh vertex indices. See the canonical face model shipped with MediaPipe.
_UPPER_INNER_LIP = 13
_LOWER_INNER_LIP = 14
_LEFT_MOUTH_CORNER = 78
_RIGHT_MOUTH_CORNER = 308
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

#: Inner lip ring, in drawing order, for the debug overlay.
INNER_LIP_CONTOUR: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)


@dataclass(frozen=True)
class MouthShape:
    """Mouth geometry for one frame, in pixels and in normalised units.

    ``aperture`` and ``width`` are divided by the inter-ocular distance, so they
    describe the shape of the mouth rather than its size on screen.
    """

    aperture: float
    width: float
    centre: np.ndarray
    contour: np.ndarray
    interocular: float

    @property
    def box(self) -> BoundingBox:
        """Tight box around the inner lips, for the on-screen indicator.

        Rounded outwards so the box always encloses the contour it came from.
        """
        x_min, y_min = np.floor(self.contour.min(axis=0))
        x_max, y_max = np.ceil(self.contour.max(axis=0))
        return BoundingBox(
            x=int(x_min),
            y=int(y_min),
            width=max(1, int(x_max - x_min)),
            height=max(1, int(y_max - y_min)),
        )


class MouthLandmarker:
    """Measures lip geometry with MediaPipe FaceMesh.

    The default confidence is below MediaPipe's own because a missed frame here
    reads as silence and suppresses an utterance, whereas a loose fit costs
    nothing: the mesh only steers segmentation, and the transcription path
    re-detects the face itself.

    An instance assumes it is being fed one continuous video: FaceMesh tracks
    from the previous frame and only re-runs detection when tracking fails, so a
    face that jumps position between calls can be missed for a frame or two.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
    ) -> None:
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> MouthLandmarker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def measure(self, image: np.ndarray) -> MouthShape | None:
        """Return the mouth geometry in ``image``, or None if no face fits.

        ``image`` must be uint8 RGB; MediaPipe reads the buffer directly.
        """
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 RGB frame, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB frame, got {image.shape}")
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        results = self._mesh.process(image)
        if not results.multi_face_landmarks:
            return None

        height, width = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        points = np.array(
            [[point.x * width, point.y * height] for point in landmarks],
            dtype=np.float32,
        )

        interocular = float(
            np.linalg.norm(points[_LEFT_EYE_OUTER] - points[_RIGHT_EYE_OUTER])
        )
        # A degenerate mesh (face edge-on, or collapsed onto a point) would make
        # the normalisation explode rather than merely be inaccurate.
        if interocular < 1.0:
            return None

        upper, lower = points[_UPPER_INNER_LIP], points[_LOWER_INNER_LIP]
        left, right = points[_LEFT_MOUTH_CORNER], points[_RIGHT_MOUTH_CORNER]
        return MouthShape(
            aperture=float(np.linalg.norm(upper - lower)) / interocular,
            width=float(np.linalg.norm(left - right)) / interocular,
            centre=(upper + lower) / 2.0,
            contour=points[list(INNER_LIP_CONTOUR)],
            interocular=interocular,
        )
