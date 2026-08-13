"""Sweep lip-motion segmentation settings against the labelled fixtures.

Not part of the application. It exists so the thresholds in
:class:`ml.config.StreamingConfig` can be justified from measurements rather
than guessed, and re-derived when the fixtures or the signal change.

Run with ``python scripts/tune_motion_thresholds.py``.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import StreamingConfig  # noqa: E402
from ml.preprocessing.mouth_landmarks import MouthLandmarker  # noqa: E402
from ml.preprocessing.video_io import decode_frames  # noqa: E402
from ml.streaming.segmenter import UtteranceSegmenter  # noqa: E402

COMPOSITE_DIR = Path("data/test_videos/composite")

#: Clips that must produce a segment, and clips that must not.
SHOULD_SEGMENT = "speech"
SHOULD_NOT_SEGMENT = "quiet"


def _fixtures() -> dict[str, tuple[str, list[np.ndarray]]]:
    from scripts.make_test_videos import _clip, _load_portrait, _talking_clip

    portrait = _load_portrait()
    fixtures: dict[str, tuple[str, list[np.ndarray]]] = {
        "talking": (SHOULD_SEGMENT, _talking_clip(90, portrait)),
        "still": (SHOULD_NOT_SEGMENT, _clip(90, "none", portrait)),
        "head_move": (SHOULD_NOT_SEGMENT, _clip(90, "translate", portrait)),
        "head_scale": (SHOULD_NOT_SEGMENT, _clip(90, "scale", portrait)),
        "head_rot": (SHOULD_NOT_SEGMENT, _clip(90, "rotate", portrait)),
    }
    for path in sorted(COMPOSITE_DIR.glob("*.mp4")):
        frames = [frame.image for frame in decode_frames(path, target_fps=25)]
        fixtures[path.stem] = (SHOULD_SEGMENT, frames)
    return fixtures


def _apertures(frames: list[np.ndarray]) -> list[float | None]:
    with MouthLandmarker() as landmarker:
        shapes = [landmarker.measure(frame) for frame in frames]
    return [None if shape is None else shape.aperture for shape in shapes]


def _energies(apertures: list[float | None], statistic: str, window: int) -> list[float]:
    recent: deque[float] = deque(maxlen=window)
    previous: float | None = None
    energies: list[float] = []
    for aperture in apertures:
        if aperture is None:
            previous = None
            recent.clear()
            energies.append(0.0)
            continue
        if previous is None:
            previous = aperture
            energies.append(0.0)
            continue
        recent.append(abs(aperture - previous))
        previous = aperture
        energies.append(max(recent) if statistic == "max" else float(np.mean(recent)))
    return energies


def _segments(energies: list[float], start: float, stop: float, fps: int = 25) -> int:
    config = StreamingConfig(motion_start_threshold=start, motion_stop_threshold=stop)
    segmenter = UtteranceSegmenter(config)
    count = 0
    for index, energy in enumerate(energies):
        if segmenter.update(energy, index / fps) is not None:
            count += 1
    if segmenter.flush((len(energies) - 1) / fps) is not None:
        count += 1
    return count


def main() -> int:
    fixtures = _fixtures()
    print("measuring aperture (one FaceMesh pass per clip)...")
    apertures = {
        name: (label, _apertures(frames)) for name, (label, frames) in fixtures.items()
    }
    for name, (label, series) in apertures.items():
        missing = sum(value is None for value in series)
        print(f"  {name:18s} {label:6s} {len(series):3d} frames, {missing:3d} unfitted")

    best: list[tuple] = []
    for statistic in ("mean", "max"):
        for window in (3, 5, 7):
            energies = {
                name: (label, _energies(series, statistic, window))
                for name, (label, series) in apertures.items()
            }
            for start in np.arange(0.008, 0.045, 0.002):
                for stop in np.arange(0.004, float(start), 0.002):
                    hits = misses = false_alarms = 0
                    for label, series in energies.values():
                        found = _segments(series, float(start), float(stop)) > 0
                        if label == SHOULD_SEGMENT:
                            hits += found
                            misses += not found
                        else:
                            false_alarms += found
                    best.append(
                        (hits - 2 * false_alarms - misses, hits, misses,
                         false_alarms, statistic, window, float(start), float(stop))
                    )

    best.sort(key=lambda row: (-row[0], row[3], -row[1]))
    print(f"\n{'score':>5} {'hit':>3} {'miss':>4} {'false':>5}  stat window  start   stop")
    for row in best[:15]:
        print(f"{row[0]:5d} {row[1]:3d} {row[2]:4d} {row[3]:5d}  {row[4]:4s} {row[5]:6d}  "
              f"{row[6]:.3f}  {row[7]:.3f}")

    config = StreamingConfig()
    print(
        f"\nper-clip at the configured setting "
        f"(start={config.motion_start_threshold}, stop={config.motion_stop_threshold}):"
    )
    for name, (label, series) in apertures.items():
        energies = _energies(series, "mean", 3)
        count = _segments(
            energies, config.motion_start_threshold, config.motion_stop_threshold
        )
        wanted = label == SHOULD_SEGMENT
        verdict = "ok " if bool(count) == wanted else "BAD"
        peaks = np.percentile([e for e in energies if e > 0] or [0.0], [50, 90])
        print(
            f"  {verdict} {name:18s} {label:6s} segments={count} "
            f"energy p50={peaks[0]:.4f} p90={peaks[1]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
