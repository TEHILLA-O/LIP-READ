"""Debug MP4 exports of what the model actually sees.

Preprocessing bugs — a crop drifting off the mouth, frames out of order, a wrong
frame rate — are close to invisible in metrics but obvious in two seconds of
video. These exports are the primary verification tool for Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.config import PreprocessingConfig
from ml.preprocessing.tensor import denormalise_to_uint8, rois_to_tensor
from ml.preprocessing.video_io import decode_frames, write_mp4
from ml.types import MouthTrack

__all__ = ["DebugExport", "export_mouth_track", "export_source_overlay"]


@dataclass(frozen=True)
class DebugExport:
    """Paths written by :func:`export_mouth_track`."""

    mouth_roi: Path
    network_input: Path
    overlay: Path | None = None

    def to_dict(self) -> dict[str, str]:
        paths = {"mouth_roi": str(self.mouth_roi), "network_input": str(self.network_input)}
        if self.overlay:
            paths["overlay"] = str(self.overlay)
        return paths


def export_mouth_track(
    track: MouthTrack,
    destination_dir: str | Path,
    stem: str = "debug",
    config: PreprocessingConfig | None = None,
) -> DebugExport:
    """Write the cropped mouth sequence and the normalised network input.

    Two files, because they answer different questions. ``*_mouth_roi.mp4`` shows
    whether alignment and cropping tracked the mouth. ``*_network_input.mp4``
    shows the 88x88 grayscale tensor after normalisation, which is the only thing
    the model receives.
    """
    config = config or PreprocessingConfig()
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    roi_path = write_mp4(
        track.rois, destination_dir / f"{stem}_mouth_roi.mp4", fps=track.target_fps
    )

    tensor = rois_to_tensor(track.rois, config=config)
    network_frames = denormalise_to_uint8(tensor, config=config)
    input_path = write_mp4(
        network_frames,
        destination_dir / f"{stem}_network_input.mp4",
        fps=track.target_fps,
    )

    return DebugExport(mouth_roi=roi_path, network_input=input_path)


def export_source_overlay(
    source_path: str | Path,
    track: MouthTrack,
    destination: str | Path,
    config: PreprocessingConfig | None = None,
) -> Path:
    """Re-render the source clip with the detected face and mouth drawn on.

    Confirms the pipeline locked onto the right speaker, which the cropped
    exports cannot show.
    """
    config = config or PreprocessingConfig()
    frames: list[np.ndarray] = []

    for frame in decode_frames(
        source_path,
        target_fps=config.target_fps,
        long_side=config.detection_long_side,
        max_frames=track.total_frames,
    ):
        image = np.ascontiguousarray(frame.image[..., ::-1])  # RGB -> BGR for cv2
        box = track.boxes[frame.index] if frame.index < len(track.boxes) else None
        if box is not None:
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 200, 255),
                2,
            )
            label = f"{frame.index:04d}"
            cv2.putText(
                image, label, (box.x, max(14, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image, "no detection (interpolated)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )
        frames.append(np.ascontiguousarray(image[..., ::-1]))  # back to RGB

    if not frames:
        raise ValueError(f"No frames decoded from {source_path}")

    # H.264 needs even dimensions and the source may be any size.
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:
        frames = [f[: height - height % 2, : width - width % 2] for f in frames]

    return write_mp4(frames, destination, fps=config.target_fps)
