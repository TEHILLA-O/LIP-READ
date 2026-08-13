"""Tile frames from a video into a single PNG.

Watching a 96x96 debug clip frame by frame is awkward; a contact sheet shows
drift, jitter and ordering problems at a glance. Intended for the mouth ROI and
network-input exports written by ``scripts/transcribe.py --debug``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.video_io import decode_frames, probe  # noqa: E402


def build_sheet(
    video: Path, columns: int = 10, max_frames: int = 40, scale: int = 2
) -> np.ndarray:
    metadata = probe(video)
    frames = [
        frame.image
        for frame in decode_frames(
            video, target_fps=int(round(metadata.fps)) or 25, long_side=None
        )
    ]
    if not frames:
        raise SystemExit(f"No frames decoded from {video}")

    step = max(1, len(frames) // max_frames)
    selected = frames[::step][:max_frames]

    height, width = selected[0].shape[:2]
    rows = (len(selected) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)

    for position, image in enumerate(selected):
        row, column = divmod(position, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
        cv2.putText(
            sheet,
            str(position * step),
            (column * width + 3, row * height + 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA,
        )

    if scale > 1:
        sheet = cv2.resize(
            sheet,
            (sheet.shape[1] * scale, sheet.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--scale", type=int, default=2)
    arguments = parser.parse_args()

    sheet = build_sheet(
        arguments.video, arguments.columns, arguments.max_frames, arguments.scale
    )
    output = arguments.output or arguments.video.with_suffix(".sheet.png")
    cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"{output}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
