"""Mean-face reference, similarity alignment and patch cutting."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.alignment import (
    MOUTH_KEYPOINT_INDEX,
    cut_patch,
    estimate_alignment,
    load_mean_face,
    stable_reference_points,
    transform_points,
    warp_face,
)


class TestMeanFace:
    def test_has_68_points(self) -> None:
        assert load_mean_face().shape == (68, 2)

    def test_lies_inside_the_reference_frame(self) -> None:
        mean_face = load_mean_face()
        assert mean_face.min() > 0
        assert mean_face.max() < 256


class TestStableReference:
    def test_returns_four_points_in_documented_order(self) -> None:
        reference = stable_reference_points()
        assert reference.shape == (4, 2)
        right_eye, left_eye, nose, mouth = reference
        assert right_eye[0] < left_eye[0]
        assert nose[1] > right_eye[1]
        assert mouth[1] > nose[1]

    def test_mouth_crop_fits_inside_the_aligned_face(self) -> None:
        """A 96x96 patch at the reference mouth must not need padding."""
        mouth = stable_reference_points(target_size=256)[MOUTH_KEYPOINT_INDEX]
        assert 48 <= mouth[0] <= 256 - 48
        assert 48 <= mouth[1] <= 256 - 48


class TestAlignment:
    @staticmethod
    def _keypoints(
        offset=(0.0, 0.0), scale: float = 1.0, rotation: float = 0.0
    ) -> np.ndarray:
        base = stable_reference_points().copy()
        centred = base - base.mean(axis=0)
        angle = np.deg2rad(rotation)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        moved = (centred * scale) @ rotation_matrix.T + base.mean(axis=0)
        return (moved + np.array(offset, dtype=np.float32)).astype(np.float32)

    def test_identity_keypoints_give_an_identity_transform(self) -> None:
        transform = estimate_alignment(self._keypoints())
        assert transform is not None
        np.testing.assert_allclose(transform[:, :2], np.eye(2), atol=1e-3)
        np.testing.assert_allclose(transform[:, 2], np.zeros(2), atol=1e-3)

    @pytest.mark.parametrize(
        "offset,scale,rotation",
        [((60, -40), 1.0, 0.0), ((0, 0), 1.7, 0.0), ((0, 0), 1.0, 25.0),
         ((-30, 25), 0.6, -15.0)],
    )
    def test_recovers_the_reference_pose(self, offset, scale, rotation) -> None:
        """Alignment must undo translation, scale and roll, which is what makes
        the mouth crop invariant to how the speaker is sitting."""
        keypoints = self._keypoints(offset, scale, rotation)
        transform = estimate_alignment(keypoints)
        aligned = transform_points(keypoints, transform)
        np.testing.assert_allclose(aligned, stable_reference_points(), atol=0.5)

    def test_rejects_the_wrong_number_of_keypoints(self) -> None:
        with pytest.raises(ValueError):
            estimate_alignment(np.zeros((5, 2), dtype=np.float32))


class TestWarpFace:
    def test_produces_a_square_canonical_face(self, portrait: np.ndarray) -> None:
        from ml.preprocessing.face_detector import FaceTracker

        with FaceTracker() as tracker:
            observation = tracker.detect(portrait)
        transform = estimate_alignment(observation.keypoints)
        aligned = warp_face(portrait, transform, target_size=256)
        assert aligned.shape == (256, 256, 3)
        assert aligned.dtype == np.uint8

    def test_places_the_mouth_at_the_reference_position(
        self, portrait: np.ndarray
    ) -> None:
        from ml.preprocessing.face_detector import FaceTracker

        with FaceTracker() as tracker:
            observation = tracker.detect(portrait)
        transform = estimate_alignment(observation.keypoints)
        aligned_points = transform_points(observation.keypoints, transform)
        expected = stable_reference_points()[MOUTH_KEYPOINT_INDEX]
        actual = aligned_points[MOUTH_KEYPOINT_INDEX]
        assert np.hypot(*(actual - expected)) < 6.0


class TestCutPatch:
    def test_returns_the_requested_size(self) -> None:
        image = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        patch = cut_patch(image, np.array([128.0, 128.0]), 48, 48)
        assert patch.shape == (96, 96)

    def test_keeps_colour_channels(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        patch = cut_patch(image, np.array([100.0, 150.0]), 48, 48)
        assert patch.shape == (96, 96, 3)

    def test_pads_rather_than_shrinking_at_an_edge(self) -> None:
        """The reference implementation raises here, aborting the whole clip."""
        image = np.full((256, 256, 3), 200, dtype=np.uint8)
        patch = cut_patch(image, np.array([5.0, 5.0]), 48, 48)
        assert patch.shape == (96, 96, 3)
        assert patch[0, 0].tolist() == [0, 0, 0]  # padded region
        assert patch[-1, -1].tolist() == [200, 200, 200]  # real image

    def test_pads_at_the_far_edge(self) -> None:
        image = np.full((256, 256), 200, dtype=np.uint8)
        patch = cut_patch(image, np.array([252.0, 252.0]), 48, 48)
        assert patch.shape == (96, 96)
        assert patch[-1, -1] == 0

    def test_is_centred_on_the_requested_point(self) -> None:
        image = np.zeros((256, 256), dtype=np.uint8)
        image[128, 100] = 255
        patch = cut_patch(image, np.array([100.0, 128.0]), 48, 48)
        assert patch[48, 48] == 255
