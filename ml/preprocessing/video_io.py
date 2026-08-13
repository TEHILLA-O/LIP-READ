"""Video decoding, frame-rate conversion and MP4 writing.

Uses PyAV, which ships its own FFmpeg libraries, so no system FFmpeg install is
required. Decoding is a generator: the upload path walks a clip twice (once to
detect faces, once to crop) rather than holding every full-resolution frame in
memory.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

__all__ = [
    "VideoMetadata",
    "probe",
    "decode_frames",
    "resample_to_fps",
    "write_mp4",
    "DecodeError",
]


class DecodeError(RuntimeError):
    """Raised when a container cannot be opened or holds no video stream."""


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    codec: str
    rotation: int


@dataclass(frozen=True)
class DecodedFrame:
    """One frame after frame-rate conversion."""

    index: int
    timestamp: float
    image: np.ndarray  # uint8 RGB, (H, W, 3)


def _rotation_of(stream: av.video.stream.VideoStream) -> int:
    """Read display rotation from container metadata.

    Phone recordings are commonly stored unrotated with a rotate tag; ignoring
    it hands the detector a sideways face.
    """
    raw = stream.metadata.get("rotate")
    if raw is not None:
        with contextlib.suppress(ValueError):
            return int(raw) % 360
    return 0


def _apply_rotation(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _scale_to_long_side(image: np.ndarray, long_side: int | None) -> np.ndarray:
    if not long_side:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image
    scale = long_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def probe(path: str | Path) -> VideoMetadata:
    """Read stream properties without decoding the whole file."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise DecodeError(f"No video stream in {path}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else 0.0
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                duration = 0.0
            rotation = _rotation_of(stream)
            width, height = stream.codec_context.width, stream.codec_context.height
            if rotation in (90, 270):
                width, height = height, width
            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                duration_seconds=duration,
                codec=stream.codec_context.name,
                rotation=rotation,
            )
    except av.AVError as exc:  # pragma: no cover - depends on file corruption
        raise DecodeError(f"Could not open {path}: {exc}") from exc


def _iter_raw_frames(
    path: str | Path, long_side: int | None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_seconds, rgb_image)`` in presentation order."""
    try:
        container = av.open(str(path))
    except av.AVError as exc:
        raise DecodeError(f"Could not open {path}: {exc}") from exc

    with container:
        if not container.streams.video:
            raise DecodeError(f"No video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rotation = _rotation_of(stream)
        time_base = stream.time_base
        fallback_period = 1.0 / float(stream.average_rate or 25)

        for position, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and time_base is not None:
                timestamp = float(frame.pts * time_base)
            else:
                timestamp = position * fallback_period
            image = frame.to_ndarray(format="rgb24")
            image = _apply_rotation(image, rotation)
            yield timestamp, _scale_to_long_side(image, long_side)


def resample_to_fps(
    frames: Iterable[tuple[float, np.ndarray]],
    target_fps: int,
    max_frames: int | None = None,
) -> Iterator[DecodedFrame]:
    """Convert an arbitrary-rate frame stream to a fixed rate.

    Picks, for each target timestamp, whichever decoded frame sits closest to
    it. Duplicates a frame when upsampling and skips frames when downsampling,
    which keeps playback timing intact — important because the model reads
    frame index as time.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    period = 1.0 / target_fps
    # Target times are computed as start + n * period rather than accumulated,
    # because repeated addition drifts by enough over a few seconds to drop the
    # final frame, and by more on long clips.
    tolerance = period * 1e-6
    emitted = 0
    start_time = 0.0
    next_time = 0.0
    previous: tuple[float, np.ndarray] | None = None

    for timestamp, image in frames:
        if previous is None:
            previous = (timestamp, image)
            start_time = next_time = timestamp
            continue
        prev_time, prev_image = previous
        while next_time <= timestamp + tolerance:
            if max_frames is not None and emitted >= max_frames:
                return
            closer_to_previous = (next_time - prev_time) <= (timestamp - next_time)
            chosen = prev_image if closer_to_previous else image
            yield DecodedFrame(index=emitted, timestamp=next_time, image=chosen)
            emitted += 1
            next_time = start_time + emitted * period
        previous = (timestamp, image)

    if previous is not None:
        # Emit any target times the loop could not reach without lookahead.
        last_time, last_image = previous
        while next_time <= last_time + tolerance:
            if max_frames is not None and emitted >= max_frames:
                return
            yield DecodedFrame(index=emitted, timestamp=next_time, image=last_image)
            emitted += 1
            next_time = start_time + emitted * period


def decode_frames(
    path: str | Path,
    target_fps: int = 25,
    long_side: int | None = 640,
    max_frames: int | None = None,
) -> Iterator[DecodedFrame]:
    """Decode ``path`` as RGB frames at exactly ``target_fps``."""
    yield from resample_to_fps(
        _iter_raw_frames(path, long_side), target_fps=target_fps, max_frames=max_frames
    )


def write_mp4(
    frames: np.ndarray | Iterable[np.ndarray],
    destination: str | Path,
    fps: int = 25,
    crf: int = 18,
) -> Path:
    """Encode a frame sequence to H.264 MP4.

    Accepts grayscale ``(T, H, W)`` or RGB ``(T, H, W, 3)`` uint8 input. Used
    for the debug export, so quality is favoured over file size: the point is to
    see exactly what the model was fed.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sequence = list(frames) if not isinstance(frames, np.ndarray) else list(frames)
    if not sequence:
        raise ValueError("Cannot write an empty video")

    first = sequence[0]
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"H.264 needs even dimensions, got {width}x{height}")

    with av.open(str(destination), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}

        for image in sequence:
            array = np.ascontiguousarray(image)
            if array.ndim == 2:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())

    return destination
