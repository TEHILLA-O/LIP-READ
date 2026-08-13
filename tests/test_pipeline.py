"""End-to-end mouth ROI extraction, independent of any model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.config import PreprocessingConfig
from ml.preprocessing.pipeline import (
    MouthROIPipeline,
    NoFaceDetectedError,
    PreprocessingError,
)


@pytest.fixture(scope="module")
def pipeline():
    with MouthROIPipeline(PreprocessingConfig()) as instance:
        yield instance


class TestFromVideo:
    def test_produces_the_documented_roi_shape(
        self, pipeline: MouthROIPipeline, still_clip: Path
    ) -> None:
        track = pipeline.from_video(still_clip)
        assert track.rois.ndim == 4
        assert track.rois.shape[1:] == (96, 96, 3)
        assert track.rois.dtype == np.uint8

    def test_detects_the_face_in_most_frames(
        self, pipeline: MouthROIPipeline, still_clip: Path
    ) -> None:
        track = pipeline.from_video(still_clip)
        assert track.detection_ratio > 0.9

    def test_metadata_is_consistent(
        self, pipeline: MouthROIPipeline, still_clip: Path
    ) -> None:
        track = pipeline.from_video(still_clip)
        assert track.total_frames == track.rois.shape[0]
        assert len(track.boxes) == track.total_frames
        assert track.aligned_keypoints.shape == (track.total_frames, 4, 2)
        assert track.target_fps == 25

    def test_raises_when_no_face_is_present(
        self, pipeline: MouthROIPipeline, no_face_clip: Path
    ) -> None:
        with pytest.raises(NoFaceDetectedError):
            pipeline.from_video(no_face_clip)

    def test_converts_frame_rate(
        self, pipeline: MouthROIPipeline, clip_60fps: Path
    ) -> None:
        track = pipeline.from_video(clip_60fps)
        assert track.target_fps == 25
        assert track.source_fps == pytest.approx(60.0, abs=0.1)
        assert track.total_frames == 50

    def test_respects_the_max_frames_cap(
        self, still_clip: Path
    ) -> None:
        from dataclasses import replace

        config = replace(PreprocessingConfig(), max_frames=10)
        with MouthROIPipeline(config) as limited:
            assert limited.from_video(still_clip).total_frames == 10


class TestAlignmentStability:
    """Alignment should leave only lip motion; head motion must be cancelled."""

    @staticmethod
    def _mouth_positions(track) -> np.ndarray:
        return track.aligned_keypoints[:, 3, :]

    def test_translation_is_cancelled(
        self, pipeline: MouthROIPipeline, moving_clip: Path
    ) -> None:
        positions = self._mouth_positions(pipeline.from_video(moving_clip))
        assert positions.std(axis=0).max() < 2.0

    def test_scale_change_is_cancelled(
        self, pipeline: MouthROIPipeline, scaling_clip: Path
    ) -> None:
        positions = self._mouth_positions(pipeline.from_video(scaling_clip))
        assert positions.std(axis=0).max() < 2.0

    def test_roll_is_cancelled(
        self, pipeline: MouthROIPipeline, rotating_clip: Path
    ) -> None:
        positions = self._mouth_positions(pipeline.from_video(rotating_clip))
        assert positions.std(axis=0).max() < 2.0

    def test_alignment_beats_a_fixed_crop_under_head_motion(
        self, pipeline: MouthROIPipeline, face_frames: list[np.ndarray]
    ) -> None:
        """Compare against the naive alternative rather than an absolute number.

        The clip contains head motion but no lip motion, so aligned crops should
        be nearly static from frame to frame while a fixed window over the same
        source is dominated by the head moving through it. Stating it as a ratio
        keeps the test meaningful regardless of codec noise.
        """
        aligned = pipeline.from_frames(face_frames).rois.astype(np.int16)
        aligned_change = np.abs(np.diff(aligned, axis=0)).mean()

        height, width = face_frames[0].shape[:2]
        top, left = height // 2 - 48, width // 2 - 48
        fixed = np.stack(
            [frame[top : top + 96, left : left + 96] for frame in face_frames]
        ).astype(np.int16)
        fixed_change = np.abs(np.diff(fixed, axis=0)).mean()

        assert aligned_change < fixed_change / 2, (
            f"alignment barely helped: aligned {aligned_change:.1f} vs "
            f"fixed crop {fixed_change:.1f}"
        )


class TestFromFrames:
    def test_matches_the_file_path_output_shape(
        self, pipeline: MouthROIPipeline, face_frames: list[np.ndarray]
    ) -> None:
        track = pipeline.from_frames(face_frames)
        assert track.total_frames == len(face_frames)
        assert track.rois.shape[1:] == (96, 96, 3)

    def test_rejects_a_clip_that_is_too_short(
        self, pipeline: MouthROIPipeline, face_frames: list[np.ndarray]
    ) -> None:
        with pytest.raises(PreprocessingError, match="at least"):
            pipeline.from_frames(face_frames[:3])

    def test_reports_the_target_frame_rate(
        self, pipeline: MouthROIPipeline, face_frames: list[np.ndarray]
    ) -> None:
        assert pipeline.from_frames(face_frames).target_fps == 25


class TestTrackProperties:
    def test_duration_matches_frame_count(
        self, pipeline: MouthROIPipeline, still_clip: Path
    ) -> None:
        track = pipeline.from_video(still_clip)
        assert track.duration_seconds == pytest.approx(track.total_frames / 25)

    def test_detection_ratio_is_bounded(
        self, pipeline: MouthROIPipeline, still_clip: Path
    ) -> None:
        assert 0.0 <= pipeline.from_video(still_clip).detection_ratio <= 1.0
