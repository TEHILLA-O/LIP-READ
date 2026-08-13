"""Face detection, keypoint layout and single-speaker tracking."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.face_detector import FaceTracker
from ml.types import BoundingBox


@pytest.fixture(scope="module")
def tracker():
    with FaceTracker() as instance:
        yield instance


class TestDetection:
    def test_finds_a_face_in_a_portrait(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        tracker.reset()
        observation = tracker.detect(portrait)
        assert observation is not None
        assert observation.score > 0.5

    def test_returns_four_keypoints(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        tracker.reset()
        observation = tracker.detect(portrait)
        assert observation.keypoints.shape == (4, 2)
        assert observation.keypoints.dtype == np.float32

    def test_keypoints_are_in_the_documented_order(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        """Alignment depends on right eye, left eye, nose tip, mouth centre."""
        tracker.reset()
        keypoints = tracker.detect(portrait).keypoints
        right_eye, left_eye, nose, mouth = keypoints

        assert right_eye[0] < left_eye[0], "right eye should be left of left eye"
        assert nose[1] > right_eye[1], "nose sits below the eyes"
        assert mouth[1] > nose[1], "mouth sits below the nose"

    def test_box_is_inside_the_frame(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        tracker.reset()
        box = tracker.detect(portrait).box
        assert box.width > 0 and box.height > 0
        assert 0 <= box.x < portrait.shape[1]
        assert 0 <= box.y < portrait.shape[0]

    def test_returns_none_when_there_is_no_face(self, tracker: FaceTracker) -> None:
        tracker.reset()
        noise = np.random.default_rng(7).integers(
            0, 255, size=(240, 320, 3), dtype=np.uint8
        )
        assert tracker.detect(noise) is None

    def test_rejects_a_float_frame(self, tracker: FaceTracker) -> None:
        with pytest.raises(TypeError):
            tracker.detect(np.zeros((64, 64, 3), dtype=np.float32))

    def test_rejects_a_grayscale_frame(self, tracker: FaceTracker) -> None:
        with pytest.raises(ValueError):
            tracker.detect(np.zeros((64, 64), dtype=np.uint8))


class TestSequenceDetection:
    def test_detects_across_a_moving_clip(
        self, tracker: FaceTracker, face_frames: list[np.ndarray]
    ) -> None:
        observations = tracker.detect_sequence(face_frames)
        found = sum(item is not None for item in observations)
        assert found / len(observations) > 0.9

    def test_frame_indices_are_recorded(
        self, tracker: FaceTracker, face_frames: list[np.ndarray]
    ) -> None:
        observations = tracker.detect_sequence(face_frames[:10])
        for index, observation in enumerate(observations):
            if observation is not None:
                assert observation.frame_index == index


class TestTracking:
    def test_stays_on_one_face_when_a_second_appears(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        """A bystander growing larger must not steal the track."""
        from scripts.make_test_videos import _clip

        frames = _clip(30, "translate", portrait, second_face=True)
        observations = tracker.detect_sequence(frames)
        centres = [
            observation.box.centre for observation in observations if observation
        ]
        assert len(centres) > 20

        # The tracked face moves smoothly; a jump to the bystander would be large.
        jumps = [
            np.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(centres, centres[1:], strict=False)
        ]
        assert max(jumps) < 80, f"track jumped between faces: max step {max(jumps):.0f}px"

    def test_reset_clears_the_track(
        self, tracker: FaceTracker, portrait: np.ndarray
    ) -> None:
        tracker.detect(portrait)
        tracker.reset()
        assert tracker._last_box is None


class TestBoundingBox:
    def test_iou_of_identical_boxes_is_one(self) -> None:
        box = BoundingBox(10, 10, 50, 50)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self) -> None:
        assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(50, 50, 10, 10)) == 0.0

    def test_iou_of_half_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 10, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_centre(self) -> None:
        assert BoundingBox(10, 20, 30, 40).centre == (25.0, 40.0)
