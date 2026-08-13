"""Landmark gap filling and temporal smoothing.

Alignment is done per frame, so raw detector jitter turns into a crop that
wobbles around the mouth. The front-end cannot tell that apart from lip motion,
which is why Auto-AVSR smooths the landmark *shape* over a short window while
keeping each frame's own centre.
"""

from __future__ import annotations

import numpy as np

__all__ = ["interpolate_missing", "smooth_landmarks", "detected_mask"]


def detected_mask(landmarks: list[np.ndarray | None]) -> np.ndarray:
    """Boolean mask of frames that carry a real detection."""
    return np.array([item is not None for item in landmarks], dtype=bool)


def interpolate_missing(landmarks: list[np.ndarray | None]) -> np.ndarray | None:
    """Fill undetected frames so every frame has landmarks.

    Interior gaps are linearly interpolated between the surrounding detections;
    leading and trailing gaps are held at the nearest detection. Returns None
    when the clip contains no detections at all.
    """
    valid = [index for index, item in enumerate(landmarks) if item is not None]
    if not valid:
        return None

    template = landmarks[valid[0]]
    filled = np.empty((len(landmarks), *template.shape), dtype=np.float32)

    for previous, following in zip(valid, valid[1:], strict=False):
        filled[previous] = landmarks[previous]
        gap = following - previous
        if gap > 1:
            start = np.asarray(landmarks[previous], dtype=np.float32)
            delta = np.asarray(landmarks[following], dtype=np.float32) - start
            for step in range(1, gap):
                filled[previous + step] = start + (step / gap) * delta
    filled[valid[-1]] = landmarks[valid[-1]]

    filled[: valid[0]] = filled[valid[0]]
    filled[valid[-1] + 1 :] = filled[valid[-1]]
    return filled


def smooth_landmarks(landmarks: np.ndarray, index: int, window: int = 12) -> np.ndarray:
    """Average landmark shape around ``index`` but keep that frame's position.

    The window shrinks at clip boundaries so the first and last frames are not
    dragged toward the interior.
    """
    if landmarks.ndim != 3:
        raise ValueError(f"Expected (T, N, 2) landmarks, got shape {landmarks.shape}")

    total = landmarks.shape[0]
    margin = min(window // 2, index, total - 1 - index)
    if margin <= 0:
        return landmarks[index].astype(np.float32, copy=True)

    neighbourhood = landmarks[index - margin : index + margin + 1]
    smoothed = neighbourhood.mean(axis=0)
    # Re-centre on this frame so smoothing removes shape jitter, not head motion.
    smoothed += landmarks[index].mean(axis=0) - smoothed.mean(axis=0)
    return smoothed.astype(np.float32, copy=False)
