"""Face alignment and mouth ROI extraction.

Every frame is warped by a similarity transform that puts the eyes, nose and
mouth on a fixed reference face, then a constant-size patch is cut at the mouth.
The alignment is what makes the crop invariant to head pose, distance and roll,
so the front-end only ever sees lip motion.

The reference geometry and crop size must match what the checkpoint was trained
with; see ``docs/preprocessing-spec.md``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ml.paths import MEAN_FACE_PATH

__all__ = [
    "MOUTH_KEYPOINT_INDEX",
    "load_mean_face",
    "stable_reference_points",
    "estimate_alignment",
    "warp_face",
    "transform_points",
    "cut_patch",
]

#: Index of the mouth-centre keypoint within the four alignment points.
MOUTH_KEYPOINT_INDEX = 3

# 68-point landmark groups used to build the four reference points.
_RIGHT_EYE = slice(36, 42)
_LEFT_EYE = slice(42, 48)
_NOSE = slice(31, 36)
_MOUTH = slice(48, 68)


@lru_cache(maxsize=1)
def load_mean_face() -> np.ndarray:
    """Load the 68-point mean face the checkpoints were aligned to."""
    if not MEAN_FACE_PATH.exists():
        raise FileNotFoundError(
            f"Mean face reference not found at {MEAN_FACE_PATH}. "
            "Run `python scripts/fetch_references.py`."
        )
    mean_face = np.load(MEAN_FACE_PATH).astype(np.float32)
    if mean_face.shape != (68, 2):
        raise ValueError(f"Expected a (68, 2) mean face, got {mean_face.shape}")
    return mean_face


@lru_cache(maxsize=4)
def stable_reference_points(
    target_size: int = 256, reference_size: int = 256
) -> np.ndarray:
    """Reference positions of right eye, left eye, nose tip and mouth centre.

    Order matches :data:`ml.preprocessing.face_detector.KEYPOINT_ORDER`.
    """
    mean_face = load_mean_face()
    reference = np.vstack(
        [
            mean_face[_RIGHT_EYE].mean(axis=0),
            mean_face[_LEFT_EYE].mean(axis=0),
            mean_face[_NOSE].mean(axis=0),
            mean_face[_MOUTH].mean(axis=0),
        ]
    ).astype(np.float32)
    reference -= (reference_size - target_size) / 2.0
    return reference


def estimate_alignment(
    keypoints: np.ndarray, target_size: int = 256
) -> np.ndarray | None:
    """Estimate the 2x3 similarity transform onto the reference face.

    Returns None when OpenCV cannot fit a transform, which happens on degenerate
    detections such as a fully profile face whose keypoints collapse onto a line.
    """
    if keypoints.shape != (4, 2):
        raise ValueError(f"Expected 4 alignment keypoints, got {keypoints.shape}")

    reference = stable_reference_points(target_size=target_size)
    transform, _ = cv2.estimateAffinePartial2D(
        keypoints.astype(np.float32), reference, method=cv2.LMEDS
    )
    return transform


def warp_face(
    image: np.ndarray, transform: np.ndarray, target_size: int = 256
) -> np.ndarray:
    """Apply the alignment transform, producing a square canonical face."""
    return cv2.warpAffine(
        image,
        transform,
        dsize=(target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map points through a 2x3 affine transform."""
    return (points @ transform[:, :2].T + transform[:, 2]).astype(np.float32)


def cut_patch(
    image: np.ndarray, centre: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Cut a ``2*height`` x ``2*width`` patch centred on ``centre``.

    Pads with zeros when the patch runs past the aligned face, matching the
    warp's own border colour. The reference implementation raises instead, which
    would abort a whole clip because of one badly aligned frame.
    """
    centre_x, centre_y = float(centre[0]), float(centre[1])
    y_min, y_max = int(round(centre_y - height)), int(round(centre_y + height))
    x_min, x_max = int(round(centre_x - width)), int(round(centre_x + width))

    pad_top = max(0, -y_min)
    pad_left = max(0, -x_min)
    pad_bottom = max(0, y_max - image.shape[0])
    pad_right = max(0, x_max - image.shape[1])

    patch = image[
        max(0, y_min) : min(image.shape[0], y_max),
        max(0, x_min) : min(image.shape[1], x_max),
    ]
    if pad_top or pad_bottom or pad_left or pad_right:
        padding = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if patch.ndim == 3:
            padding.append((0, 0))
        patch = np.pad(patch, padding, mode="constant", constant_values=0)
    return patch
