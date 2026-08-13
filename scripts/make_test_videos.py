"""Generate deterministic synthetic test clips.

The preprocessing pipeline needs fixtures that exercise face detection,
tracking, alignment and frame-rate conversion, and it needs them to be
reproducible and free of anyone's real face. These clips composite
scikit-image's bundled public-domain portrait (NASA photograph of astronaut
Eileen Collins) onto a moving background.

They contain no lip motion, so they are useless for checking transcripts. Use
``scripts/fetch_lrs3_samples.py`` for that. What they do check is geometry:
whether the face is found, whether the crop stays locked to the mouth while the
head moves, and whether frames stay in order at every frame rate.

Every clip carries a frame-index marker: a white block in the top-left whose
horizontal position advances one step per frame, far from the face so it cannot
contaminate the mouth ROI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import TEST_VIDEO_DIR  # noqa: E402
from ml.preprocessing.video_io import write_mp4  # noqa: E402

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MARKER_SIZE = 8
MARKER_STEP = 4


def _load_portrait() -> np.ndarray:
    from skimage import data

    return np.ascontiguousarray(data.astronaut())


def _background(index: int) -> np.ndarray:
    """A slowly shifting gradient, so frames are never byte-identical."""
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    column = np.linspace(30, 90, FRAME_WIDTH, dtype=np.float32)
    canvas[:, :, 0] = np.clip(column + (index % 20), 0, 255).astype(np.uint8)
    canvas[:, :, 1] = 40
    canvas[:, :, 2] = np.clip(120 - column * 0.5, 0, 255).astype(np.uint8)
    return canvas


def _stamp_marker(frame: np.ndarray, index: int) -> None:
    """Encode the frame index as a marker position, for ordering assertions."""
    x = MARKER_STEP + (index * MARKER_STEP) % (FRAME_WIDTH // 2 - MARKER_SIZE)
    frame[4 : 4 + MARKER_SIZE, x : x + MARKER_SIZE] = 255


def read_marker_position(frame: np.ndarray) -> int:
    """Recover the marker's x position from a decoded frame.

    Tolerant of H.264 compression: looks for the brightest column in the marker
    strip rather than an exact white value.
    """
    strip = frame[4 : 4 + MARKER_SIZE, : FRAME_WIDTH // 2]
    intensity = strip.mean(axis=(0, 2)) if strip.ndim == 3 else strip.mean(axis=0)
    return int(np.argmax(intensity))


def _paste_face(
    frame: np.ndarray,
    portrait: np.ndarray,
    centre: tuple[int, int],
    scale: float,
    rotation_degrees: float = 0.0,
) -> None:
    size = max(32, int(round(portrait.shape[0] * scale)))
    face = cv2.resize(portrait, (size, size), interpolation=cv2.INTER_AREA)

    if rotation_degrees:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rotation_degrees, 1.0)
        face = cv2.warpAffine(
            face, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE
        )

    x = int(centre[0] - size / 2)
    y = int(centre[1] - size / 2)
    x_start, y_start = max(0, x), max(0, y)
    x_end = min(frame.shape[1], x + size)
    y_end = min(frame.shape[0], y + size)
    if x_end <= x_start or y_end <= y_start:
        return
    frame[y_start:y_end, x_start:x_end] = face[
        y_start - y : y_end - y, x_start - x : x_end - x
    ]


def _clip(
    frames_count: int,
    motion: str,
    portrait: np.ndarray,
    second_face: bool = False,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(frames_count):
        progress = index / max(1, frames_count - 1)
        frame = _background(index)

        centre_x, centre_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        scale, rotation = 0.62, 0.0

        if motion == "translate":
            centre_x += int(90 * np.sin(progress * 2 * np.pi))
            centre_y += int(40 * np.cos(progress * 2 * np.pi))
        elif motion == "scale":
            scale = 0.45 + 0.30 * (0.5 + 0.5 * np.sin(progress * 2 * np.pi))
        elif motion == "rotate":
            rotation = 18.0 * np.sin(progress * 2 * np.pi)

        if second_face:
            # A bystander that grows larger than the tracked speaker part-way
            # through, so tracker lock-on can be tested.
            _paste_face(frame, portrait, (110, 150), 0.30 + 0.35 * progress)

        _paste_face(frame, portrait, (centre_x, centre_y), scale, rotation)
        _stamp_marker(frame, index)
        frames.append(frame)
    return frames


def _talking_clip(frames_count: int, portrait: np.ndarray) -> list[np.ndarray]:
    """A still face with an opening and closing mouth.

    Not speech, and useless for checking transcripts. Its job is to produce the
    lip-motion energy the live segmenter keys on, so that utterance detection
    and the streaming path can be tested without recording anyone. The mouth
    position is found once with the real detector rather than guessed, so the
    animation lands where the pipeline will look for it.
    """
    from ml.preprocessing.face_detector import FaceTracker

    base = _clip(frames_count, "none", portrait)
    with FaceTracker() as tracker:
        observation = tracker.detect(base[0])
    if observation is None:
        raise RuntimeError("Could not locate the mouth to animate")

    centre = observation.keypoints[3]
    width = max(6, int(observation.box.width * 0.22))

    frames = []
    for index, frame in enumerate(base):
        # Two-thirds of a cycle per frame keeps successive frames far apart, so
        # the energy signal stays well above the segmenter's start threshold.
        openness = 0.5 + 0.5 * np.sin(index * 2.0)
        height = max(2, int(width * (0.15 + 0.85 * openness)))
        cv2.ellipse(
            frame,
            (int(centre[0]), int(centre[1])),
            (width, height),
            0,
            0,
            360,
            (35, 20, 25),
            -1,
        )
        frames.append(frame)
    return frames


def _noise_clip(frames_count: int) -> list[np.ndarray]:
    generator = np.random.default_rng(seed=1234)
    frames = []
    for index in range(frames_count):
        frame = _background(index)
        noise = generator.integers(0, 60, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _stamp_marker(frame, index)
        frames.append(frame)
    return frames


def build(output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    portrait = _load_portrait()

    specifications: list[dict[str, object]] = [
        {"name": "face_still", "frames": 75, "fps": 25, "motion": "none",
         "purpose": "Baseline detection and a stable 96x96 crop."},
        {"name": "face_translate", "frames": 75, "fps": 25, "motion": "translate",
         "purpose": "Alignment must cancel head translation."},
        {"name": "face_scale", "frames": 75, "fps": 25, "motion": "scale",
         "purpose": "Alignment must cancel distance changes."},
        {"name": "face_rotate", "frames": 75, "fps": 25, "motion": "rotate",
         "purpose": "Similarity transform must cancel in-plane head roll."},
        {"name": "face_30fps", "frames": 90, "fps": 30, "motion": "translate",
         "purpose": "Downsampling 30 -> 25 fps."},
        {"name": "face_60fps", "frames": 180, "fps": 60, "motion": "translate",
         "purpose": "Downsampling 60 -> 25 fps."},
        {"name": "face_15fps", "frames": 30, "fps": 15, "motion": "translate",
         "purpose": "Upsampling 15 -> 25 fps."},
        {"name": "two_faces", "frames": 75, "fps": 25, "motion": "translate",
         "second_face": True,
         "purpose": "Tracker must stay on one speaker when a second face grows."},
        {"name": "face_talking", "frames": 75, "fps": 25, "motion": "talking",
         "purpose": "Animated mouth; drives live utterance segmentation."},
        {"name": "face_short", "frames": 4, "fps": 25, "motion": "none",
         "purpose": "Clip below the minimum usable length."},
        {"name": "no_face", "frames": 50, "fps": 25, "motion": "noise",
         "purpose": "No detectable face; the pipeline must say so."},
    ]

    manifest: list[dict[str, object]] = []
    for specification in specifications:
        name = str(specification["name"])
        count = int(specification["frames"])
        fps = int(specification["fps"])
        motion = str(specification["motion"])

        if motion == "noise":
            frames = _noise_clip(count)
        elif motion == "talking":
            frames = _talking_clip(count, portrait)
        else:
            frames = _clip(
                count, motion, portrait, bool(specification.get("second_face", False))
            )

        path = output_dir / f"{name}.mp4"
        write_mp4(frames, path, fps=fps)
        entry = {
            "name": name,
            "file": path.name,
            "frames": count,
            "fps": fps,
            "seconds": round(count / fps, 2),
            "expected_frames_at_25fps": max(1, int(count / fps * 25)),
            "has_face": motion != "noise",
            "purpose": specification["purpose"],
        }
        manifest.append(entry)
        print(f"{name:16s} {count:4d} frames @ {fps:3d} fps  ->  {path.name}")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(manifest)} clips to {output_dir}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TEST_VIDEO_DIR / "synthetic")
    arguments = parser.parse_args()
    build(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
