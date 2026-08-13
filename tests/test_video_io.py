"""Decoding, frame-rate conversion, frame ordering and MP4 writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.preprocessing.video_io import (
    DecodeError,
    decode_frames,
    probe,
    resample_to_fps,
    write_mp4,
)
from scripts.make_test_videos import read_marker_position


class TestProbe:
    def test_reads_stream_properties(self, still_clip: Path) -> None:
        metadata = probe(still_clip)
        assert (metadata.width, metadata.height) == (640, 480)
        assert metadata.fps == pytest.approx(25.0, abs=0.01)
        assert metadata.codec == "h264"
        assert metadata.rotation == 0

    def test_rejects_a_non_video_file(self, tmp_path: Path) -> None:
        path = tmp_path / "not-a-video.mp4"
        path.write_bytes(b"definitely not a container")
        with pytest.raises(DecodeError):
            probe(path)


class TestDecoding:
    def test_yields_contiguous_rgb_frames(self, still_clip: Path) -> None:
        frames = list(decode_frames(still_clip, target_fps=25, long_side=None))
        assert frames
        first = frames[0]
        assert first.image.dtype == np.uint8
        assert first.image.shape == (480, 640, 3)

    def test_indices_are_sequential(self, moving_clip: Path) -> None:
        frames = list(decode_frames(moving_clip, target_fps=25, long_side=None))
        assert [frame.index for frame in frames] == list(range(len(frames)))

    def test_timestamps_increase_monotonically(self, moving_clip: Path) -> None:
        timestamps = [f.timestamp for f in decode_frames(moving_clip, target_fps=25)]
        assert all(b > a for a, b in zip(timestamps, timestamps[1:], strict=False))

    def test_frames_stay_in_source_order(self, moving_clip: Path) -> None:
        """The marker's position advances one step per source frame."""
        positions = [
            read_marker_position(frame.image)
            for frame in decode_frames(moving_clip, target_fps=25, long_side=None)
        ]
        # The marker wraps around, so check ordering within each run.
        increases = sum(
            1 for a, b in zip(positions, positions[1:], strict=False) if b > a
        )
        assert increases >= len(positions) - 2

    def test_long_side_scaling_preserves_aspect_ratio(self, still_clip: Path) -> None:
        frame = next(iter(decode_frames(still_clip, target_fps=25, long_side=320)))
        assert max(frame.image.shape[:2]) == 320
        assert frame.image.shape[:2] == (240, 320)

    def test_max_frames_is_respected(self, moving_clip: Path) -> None:
        frames = list(decode_frames(moving_clip, target_fps=25, max_frames=7))
        assert len(frames) == 7


class TestFrameRateConversion:
    """The model reads frame index as time, so rate conversion must be exact."""

    @staticmethod
    def _synthetic(count: int, fps: float) -> list[tuple[float, np.ndarray]]:
        return [
            (index / fps, np.full((4, 4, 3), index % 256, dtype=np.uint8))
            for index in range(count)
        ]

    def test_downsamples_60_to_25(self) -> None:
        source = self._synthetic(120, 60.0)  # 1.983 s
        result = list(resample_to_fps(source, target_fps=25))
        assert len(result) == 50
        assert result[0].timestamp == pytest.approx(0.0)
        assert result[-1].timestamp == pytest.approx(49 / 25)

    def test_upsamples_15_to_25(self) -> None:
        source = self._synthetic(30, 15.0)  # 1.933 s
        result = list(resample_to_fps(source, target_fps=25))
        assert len(result) == 49

    def test_identity_when_rates_match(self) -> None:
        source = self._synthetic(50, 25.0)
        result = list(resample_to_fps(source, target_fps=25))
        assert len(result) == 50

    def test_no_cumulative_drift_over_a_long_clip(self) -> None:
        """Accumulating the period instead of multiplying loses frames."""
        source = self._synthetic(25 * 120, 25.0)  # two minutes
        result = list(resample_to_fps(source, target_fps=25))
        assert len(result) == 25 * 120
        assert result[-1].timestamp == pytest.approx((25 * 120 - 1) / 25, abs=1e-6)

    def test_picks_the_nearest_frame_in_time(self) -> None:
        source = self._synthetic(60, 30.0)
        result = list(resample_to_fps(source, target_fps=25))
        # Target 0.04s sits between source frames at 0.0333 and 0.0667.
        assert result[1].image[0, 0, 0] == 1

    def test_rejects_a_non_positive_rate(self) -> None:
        with pytest.raises(ValueError):
            list(resample_to_fps(self._synthetic(5, 25.0), target_fps=0))

    @pytest.mark.parametrize(
        "fixture_name,expected",
        [("clip_30fps", 50), ("clip_60fps", 50), ("clip_15fps", 49)],
    )
    def test_real_files_convert_to_25fps(
        self, request: pytest.FixtureRequest, fixture_name: str, expected: int
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        frames = list(decode_frames(path, target_fps=25))
        assert len(frames) == expected


class TestWriteMp4:
    def test_writes_playable_grayscale_video(self, tmp_path: Path) -> None:
        frames = np.random.default_rng(0).integers(
            0, 255, size=(20, 88, 88), dtype=np.uint8
        )
        path = write_mp4(frames, tmp_path / "grey.mp4", fps=25)
        assert path.exists() and path.stat().st_size > 0

        decoded = list(decode_frames(path, target_fps=25, long_side=None))
        assert len(decoded) == 20
        assert decoded[0].image.shape == (88, 88, 3)

    def test_writes_rgb_video(self, tmp_path: Path) -> None:
        frames = np.random.default_rng(1).integers(
            0, 255, size=(10, 96, 96, 3), dtype=np.uint8
        )
        path = write_mp4(frames, tmp_path / "rgb.mp4", fps=25)
        assert len(list(decode_frames(path, target_fps=25, long_side=None))) == 10

    def test_rejects_odd_dimensions(self, tmp_path: Path) -> None:
        frames = np.zeros((5, 87, 87), dtype=np.uint8)
        with pytest.raises(ValueError, match="even dimensions"):
            write_mp4(frames, tmp_path / "odd.mp4")

    def test_rejects_an_empty_sequence(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            write_mp4([], tmp_path / "empty.mp4")
