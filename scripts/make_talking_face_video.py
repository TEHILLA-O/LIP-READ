"""Build a full-scene talking-face video with a known transcript.

The synthetic clips in ``make_test_videos.py`` have a face but no lip motion, so
they can only test geometry. The LRS3 samples have real lip motion but arrive
already cropped to 96x96, so they bypass detection and alignment entirely.
Neither can answer the question that matters for Phase 1: does the *whole* chain,
from an ordinary video frame to a transcript, work?

This script makes a clip that can. It composites an LRS3 mouth sequence back
into a full scene by running the alignment transform backwards:

    LRS3 96x96 crop -> paste at the mouth of the canonical 256x256 face
                    -> warp back through the inverse alignment
                    -> full 640x480 frame with a face in it

Run the pipeline forwards on the result and it should recover very nearly the
original crop, and therefore the original transcript. Any large gap is a real
preprocessing bug, not a modelling one.

This is a test fixture, not training data. The mouth belongs to a different
person than the face around it, and the seam is visible if you look for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import PreprocessingConfig  # noqa: E402
from ml.paths import TEST_VIDEO_DIR  # noqa: E402
from ml.preprocessing.alignment import (  # noqa: E402
    MOUTH_KEYPOINT_INDEX,
    estimate_alignment,
    transform_points,
    warp_face,
)
from ml.preprocessing.face_detector import FaceTracker  # noqa: E402
from ml.preprocessing.video_io import write_mp4  # noqa: E402
from scripts.make_test_videos import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    _background,
    _load_portrait,
    _paste_face,
    _stamp_marker,
)

LRS3_DIR = TEST_VIDEO_DIR / "lrs3"
OUTPUT_DIR = TEST_VIDEO_DIR / "composite"

#: Width of the blend at the edge of the pasted mouth, in canonical pixels. Wide
#: enough that the detector never sees a hard rectangle, narrow enough that the
#: middle of the crop is untouched.
_FEATHER = 8


def _feather_mask(size: int, border: int) -> np.ndarray:
    """A square mask that fades linearly to zero at the edges."""
    ramp = np.ones(size, dtype=np.float32)
    edge = np.linspace(0.0, 1.0, border + 2, dtype=np.float32)[1:-1]
    ramp[:border] = edge
    ramp[-border:] = edge[::-1]
    return np.minimum(ramp[:, None], ramp[None, :])


def _scene_frame(index: int, portrait: np.ndarray, motion: str) -> np.ndarray:
    frame = _background(index)
    centre_x, centre_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    if motion == "translate":
        # Deliberately gentle. Violent motion tests the tracker, but here the
        # point is to measure transcription, so the head must stay legible.
        centre_x += int(35 * np.sin(index / 25.0 * 2 * np.pi))
        centre_y += int(15 * np.cos(index / 25.0 * 2 * np.pi))
    _paste_face(frame, portrait, (centre_x, centre_y), 0.62)
    _stamp_marker(frame, index)
    return frame


def _implant_mouth(
    frame: np.ndarray,
    mouth: np.ndarray,
    tracker: FaceTracker,
    config: PreprocessingConfig,
) -> np.ndarray | None:
    """Replace the face's mouth with ``mouth``, in canonical alignment space."""
    observation = tracker.detect(frame)
    if observation is None:
        return None

    transform = estimate_alignment(
        observation.keypoints, target_size=config.aligned_face_size
    )
    if transform is None:
        return None

    aligned = warp_face(frame, transform, target_size=config.aligned_face_size)
    points = transform_points(observation.keypoints, transform)
    centre_x, centre_y = points[MOUTH_KEYPOINT_INDEX]

    half_h, half_w = config.crop_height // 2, config.crop_width // 2
    top, left = int(round(centre_y)) - half_h, int(round(centre_x)) - half_w
    bottom, right = top + config.crop_height, left + config.crop_width
    canvas_size = config.aligned_face_size
    if top < 0 or left < 0 or bottom > canvas_size or right > canvas_size:
        return None

    patch = cv2.cvtColor(mouth, cv2.COLOR_GRAY2RGB) if mouth.ndim == 2 else mouth
    if patch.shape[:2] != (config.crop_height, config.crop_width):
        patch = cv2.resize(
            patch, (config.crop_width, config.crop_height), interpolation=cv2.INTER_AREA
        )

    # Match the surrounding skin's brightness so the graft is not a bright square
    # that the detector reads as a highlight.
    target = aligned[top:bottom, left:right].astype(np.float32)
    patch = patch.astype(np.float32)
    patch += target.mean() - patch.mean()

    alpha = _feather_mask(config.crop_height, _FEATHER)[..., None]
    blended = np.zeros_like(aligned, dtype=np.float32)
    blended[top:bottom, left:right] = np.clip(patch, 0, 255)

    mask = np.zeros((canvas_size, canvas_size, 1), dtype=np.float32)
    mask[top:bottom, left:right] = alpha

    height, width = frame.shape[:2]
    warp_back = dict(
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT
    )
    source_patch = cv2.warpAffine(blended, transform, (width, height), **warp_back)
    source_mask = cv2.warpAffine(mask, transform, (width, height), **warp_back)
    if source_mask.ndim == 2:
        source_mask = source_mask[..., None]

    composited = frame.astype(np.float32) * (1 - source_mask) + source_patch * source_mask
    return np.clip(composited, 0, 255).astype(np.uint8)


def build_clip(
    sample: str, motion: str, config: PreprocessingConfig
) -> tuple[list[np.ndarray], str]:
    mouth_frames = np.load(LRS3_DIR / f"{sample}.npz")["frames"]
    transcript = (LRS3_DIR / f"{sample}.txt").read_text(encoding="utf-8").strip()
    portrait = _load_portrait()

    frames: list[np.ndarray] = []
    with FaceTracker() as tracker:
        for index, mouth in enumerate(mouth_frames):
            scene = _scene_frame(index, portrait, motion)
            composited = _implant_mouth(scene, mouth, tracker, config)
            if composited is None:
                raise RuntimeError(
                    f"Could not place the mouth on frame {index}; the generated "
                    "face was not detected or could not be aligned."
                )
            frames.append(composited)
    return frames, transcript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="*", default=None, help="LRS3 sample stems")
    parser.add_argument(
        "--motion", choices=["none", "translate"], default="none",
        help="Head motion to add; 'translate' also exercises tracking",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    if not LRS3_DIR.exists():
        print(
            f"No LRS3 samples in {LRS3_DIR}. Run scripts/fetch_lrs3_samples.py first.",
            file=sys.stderr,
        )
        return 1

    samples = arguments.samples or sorted(
        path.stem for path in LRS3_DIR.glob("sample_*.npz")
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    config = PreprocessingConfig()

    manifest: list[dict[str, object]] = []
    for sample in samples:
        frames, transcript = build_clip(sample, arguments.motion, config)
        path = arguments.output / f"{sample}_{arguments.motion}.mp4"
        write_mp4(frames, path, fps=config.target_fps)
        (path.with_suffix(".txt")).write_text(transcript, encoding="utf-8")

        manifest.append({
            "file": path.name,
            "source_sample": sample,
            "motion": arguments.motion,
            "frames": len(frames),
            "fps": config.target_fps,
            "transcript": transcript,
        })
        print(f"{path.name:34s} {len(frames):3d} frames  {transcript!r}")

    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(manifest)} clip(s) to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
