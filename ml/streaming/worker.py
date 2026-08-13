"""Asynchronous inference queue for the live path.

The WebSocket loop must never wait for a 250M-parameter model. Segments are
handed to this worker, which runs them on a thread and publishes results back.

The queue is intentionally shallow and drops work rather than growing. On a
stream, a backlog is worthless: by the time a queued segment is transcribed the
speaker has moved on, and every queued job pushes the *next* result further
behind. Dropping keeps latency bounded at the cost of skipping utterances, and
the drop is reported so the UI can say so.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ml.config import StreamingConfig
from ml.inference.engine import InferenceEngine
from ml.types import TranscriptionResult

logger = logging.getLogger(__name__)

__all__ = ["InferenceWorker", "InferenceJob", "SubmitOutcome", "JobResult"]


class SubmitOutcome(str, Enum):
    QUEUED = "queued"
    #: The queue was full, so the oldest waiting job was discarded to make room.
    REPLACED_OLDEST = "replaced_oldest"
    #: The worker is not running.
    REJECTED = "rejected"


@dataclass
class InferenceJob:
    sequence: int
    frames: list[np.ndarray]
    created_at: float = field(default_factory=time.monotonic)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


@dataclass
class JobResult:
    sequence: int
    result: TranscriptionResult
    queue_wait_ms: int
    dropped: bool = False


class InferenceWorker:
    """Runs one segment at a time, dropping anything that falls behind."""

    def __init__(
        self,
        engine: InferenceEngine,
        config: StreamingConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or StreamingConfig()
        self._pending: deque[InferenceJob] = deque()
        self._wakeup = asyncio.Event()
        self._results: asyncio.Queue[JobResult] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False
        self._sequence = 0
        self.jobs_submitted = 0
        self.jobs_dropped = 0
        self.jobs_completed = 0

    # ------------------------------------------------------------------ control

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="vsr-inference-worker")

    async def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._pending.clear()

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------- submit

    def submit(self, frames: Sequence[np.ndarray]) -> tuple[SubmitOutcome, int]:
        """Queue a segment. Returns the outcome and its sequence number."""
        if not self._running:
            return SubmitOutcome.REJECTED, -1

        self._sequence += 1
        job = InferenceJob(sequence=self._sequence, frames=list(frames))

        outcome = SubmitOutcome.QUEUED
        while len(self._pending) >= self.config.max_queued_jobs:
            discarded = self._pending.popleft()
            self.jobs_dropped += 1
            outcome = SubmitOutcome.REPLACED_OLDEST
            logger.debug(
                "Dropped queued segment %d (%.1fs old) to stay current",
                discarded.sequence,
                discarded.age_seconds,
            )

        self._pending.append(job)
        self.jobs_submitted += 1
        self._wakeup.set()
        return outcome, job.sequence

    async def results(self):
        """Async iterator over completed results."""
        while self._running or not self._results.empty():
            try:
                yield await asyncio.wait_for(self._results.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

    def try_get_result(self) -> JobResult | None:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -------------------------------------------------------------------- loop

    async def _run(self) -> None:
        while self._running:
            if not self._pending:
                self._wakeup.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), timeout=0.5)
                continue

            job = self._pending.popleft()

            # A job that waited too long describes speech the user has already
            # finished; transcribing it would only display stale text.
            if job.age_seconds > self.config.max_frame_age_seconds:
                self.jobs_dropped += 1
                logger.info(
                    "Discarded segment %d: %.1fs stale", job.sequence, job.age_seconds
                )
                await self._results.put(
                    JobResult(
                        sequence=job.sequence,
                        result=TranscriptionResult.failure(
                            "Skipped: the system fell behind the camera.",
                            frames_processed=len(job.frames),
                        ),
                        queue_wait_ms=int(job.age_seconds * 1000),
                        dropped=True,
                    )
                )
                continue

            queue_wait_ms = int(job.age_seconds * 1000)
            try:
                result, _ = await asyncio.to_thread(
                    self.engine.transcribe_frames, job.frames
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the stream
                logger.exception("Inference failed for segment %d", job.sequence)
                result = TranscriptionResult.failure(
                    f"Inference failed: {type(exc).__name__}",
                    frames_processed=len(job.frames),
                )

            self.jobs_completed += 1
            await self._results.put(
                JobResult(
                    sequence=job.sequence, result=result, queue_wait_ms=queue_wait_ms
                )
            )

    def stats(self) -> dict[str, int]:
        return {
            "submitted": self.jobs_submitted,
            "completed": self.jobs_completed,
            "dropped": self.jobs_dropped,
            "queue_depth": self.queue_depth,
        }
