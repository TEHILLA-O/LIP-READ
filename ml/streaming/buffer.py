"""Rolling frame buffer for live capture.

Visual speech is only readable over a sequence, so the live path accumulates
frames and infers over windows. The buffer is bounded: once full it discards the
oldest frame, which keeps memory flat and guarantees that what it holds is
always the most recent few seconds rather than a growing backlog.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = ["RollingFrameBuffer", "BufferedFrame"]


@dataclass(frozen=True)
class BufferedFrame:
    image: np.ndarray
    timestamp: float
    index: int


class RollingFrameBuffer:
    """Thread-safe fixed-duration frame buffer.

    The producer is a WebSocket reader and the consumer is an inference worker,
    so every operation takes a lock. Snapshots copy the frame list but not the
    frames themselves; frames are treated as immutable once appended.
    """

    def __init__(self, capacity_seconds: float = 8.0, fps: int = 25) -> None:
        if capacity_seconds <= 0 or fps <= 0:
            raise ValueError("capacity_seconds and fps must be positive")
        self.fps = fps
        self.capacity = max(1, int(round(capacity_seconds * fps)))
        self._frames: deque[BufferedFrame] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self._counter = 0
        self._dropped = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def dropped_frames(self) -> int:
        """Frames evicted before any consumer read them."""
        return self._dropped

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return len(self._frames) / self.fps

    def append(self, image: np.ndarray, timestamp: float | None = None) -> BufferedFrame:
        """Add a frame, evicting the oldest if the buffer is full."""
        with self._lock:
            if len(self._frames) == self.capacity:
                self._dropped += 1
            frame = BufferedFrame(
                image=image,
                timestamp=time.monotonic() if timestamp is None else timestamp,
                index=self._counter,
            )
            self._counter += 1
            self._frames.append(frame)
            return frame

    def latest(self, count: int) -> list[BufferedFrame]:
        """Most recent ``count`` frames, oldest first."""
        with self._lock:
            if count <= 0:
                return []
            return list(self._frames)[-count:]

    def window(self, seconds: float) -> list[BufferedFrame]:
        """Most recent ``seconds`` of frames, oldest first."""
        return self.latest(int(round(seconds * self.fps)))

    def drain(self) -> list[BufferedFrame]:
        """Take everything and empty the buffer."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    def consume(self, count: int) -> list[BufferedFrame]:
        """Remove and return the oldest ``count`` frames."""
        with self._lock:
            taken = [self._frames.popleft() for _ in range(min(count, len(self._frames)))]
            return taken

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    @staticmethod
    def images(frames: list[BufferedFrame]) -> list[np.ndarray]:
        return [frame.image for frame in frames]
