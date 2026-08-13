"""Landmark gap filling and temporal smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.landmarks import (
    detected_mask,
    interpolate_missing,
    smooth_landmarks,
)


def _points(value: float) -> np.ndarray:
    return np.full((4, 2), value, dtype=np.float32)


class TestInterpolateMissing:
    def test_returns_none_when_nothing_was_detected(self) -> None:
        assert interpolate_missing([None, None, None]) is None

    def test_leaves_a_complete_sequence_unchanged(self) -> None:
        landmarks = [_points(0), _points(1), _points(2)]
        filled = interpolate_missing(landmarks)
        np.testing.assert_allclose(filled, np.stack(landmarks))

    def test_interpolates_an_interior_gap_linearly(self) -> None:
        filled = interpolate_missing([_points(0), None, None, _points(3)])
        assert filled.shape == (4, 4, 2)
        np.testing.assert_allclose(filled[1], _points(1), atol=1e-5)
        np.testing.assert_allclose(filled[2], _points(2), atol=1e-5)

    def test_holds_leading_gaps_at_the_first_detection(self) -> None:
        filled = interpolate_missing([None, None, _points(5), _points(6)])
        np.testing.assert_allclose(filled[0], _points(5))
        np.testing.assert_allclose(filled[1], _points(5))

    def test_holds_trailing_gaps_at_the_last_detection(self) -> None:
        filled = interpolate_missing([_points(1), _points(2), None, None])
        np.testing.assert_allclose(filled[2], _points(2))
        np.testing.assert_allclose(filled[3], _points(2))

    def test_output_is_float32(self) -> None:
        assert interpolate_missing([_points(1), None, _points(3)]).dtype == np.float32

    def test_every_frame_is_filled(self) -> None:
        landmarks = [_points(0), None, _points(2), None, None, _points(5)]
        filled = interpolate_missing(landmarks)
        assert filled.shape[0] == len(landmarks)
        assert np.isfinite(filled).all()


class TestDetectedMask:
    def test_marks_present_frames(self) -> None:
        mask = detected_mask([_points(0), None, _points(2)])
        assert mask.tolist() == [True, False, True]


class TestSmoothLandmarks:
    def test_boundary_frames_are_returned_unchanged(self) -> None:
        landmarks = np.stack([_points(index) for index in range(10)])
        np.testing.assert_allclose(smooth_landmarks(landmarks, 0), landmarks[0])
        np.testing.assert_allclose(smooth_landmarks(landmarks, 9), landmarks[9])

    def test_preserves_the_frame_centroid(self) -> None:
        """Smoothing removes shape jitter; it must not drag the crop position."""
        generator = np.random.default_rng(3)
        landmarks = np.stack(
            [_points(index) + generator.normal(0, 2, (4, 2)) for index in range(20)]
        ).astype(np.float32)
        smoothed = smooth_landmarks(landmarks, 10, window=12)
        np.testing.assert_allclose(
            smoothed.mean(axis=0), landmarks[10].mean(axis=0), atol=1e-4
        )

    def test_reduces_shape_jitter(self) -> None:
        generator = np.random.default_rng(11)
        base = stable = np.array(
            [[0.0, 0.0], [40.0, 0.0], [20.0, 20.0], [20.0, 40.0]], dtype=np.float32
        )
        noisy = np.stack(
            [base + generator.normal(0, 3, (4, 2)) for _ in range(21)]
        ).astype(np.float32)

        raw_error = np.abs(noisy[10] - noisy[10].mean(axis=0) - (stable - stable.mean(axis=0))).mean()
        smoothed = smooth_landmarks(noisy, 10, window=12)
        smooth_error = np.abs(
            smoothed - smoothed.mean(axis=0) - (stable - stable.mean(axis=0))
        ).mean()
        assert smooth_error < raw_error

    def test_rejects_the_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            smooth_landmarks(np.zeros((4, 2), dtype=np.float32), 0)

    def test_window_shrinks_near_the_start(self) -> None:
        landmarks = np.stack([_points(index) for index in range(30)])
        # Frame 1 can only average over +/-1, so it stays close to its own value.
        smoothed = smooth_landmarks(landmarks, 1, window=12)
        np.testing.assert_allclose(smoothed, _points(1), atol=1e-4)
