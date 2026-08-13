"""Inference queue behaviour.

The point of the worker is that it stays current under load. These tests use a
fake engine with a controllable delay so the drop policy can be verified without
running a model.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from ml.config import StreamingConfig
from ml.streaming.worker import InferenceWorker, SubmitOutcome
from ml.types import TranscriptionResult

pytestmark = pytest.mark.asyncio


class FakeEngine:
    """Stands in for InferenceEngine, blocking until released."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[int] = []
        self.gate = threading.Event()
        self.gate.set()

    def transcribe_frames(self, frames):
        self.gate.wait(timeout=5.0)
        if self.delay:
            threading.Event().wait(self.delay)
        self.calls.append(len(frames))
        return TranscriptionResult(text=f"segment of {len(frames)}", confidence=0.5), None


def _frames(count: int = 3) -> list[np.ndarray]:
    return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(count)]


async def _drain_one(worker: InferenceWorker, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = worker.try_get_result()
        if result is not None:
            return result
        await asyncio.sleep(0.02)
    raise AssertionError("No result arrived before the timeout")


class TestSubmission:
    async def test_rejects_work_before_start(self) -> None:
        worker = InferenceWorker(FakeEngine(), StreamingConfig())
        outcome, sequence = worker.submit(_frames())
        assert outcome is SubmitOutcome.REJECTED
        assert sequence == -1

    async def test_transcribes_a_submitted_segment(self) -> None:
        engine = FakeEngine()
        worker = InferenceWorker(engine, StreamingConfig())
        await worker.start()
        try:
            _, sequence = worker.submit(_frames(4))
            job = await _drain_one(worker)
        finally:
            await worker.stop()
        assert job.sequence == sequence
        assert job.result.text == "segment of 4"
        assert engine.calls == [4]

    async def test_sequence_numbers_increase(self) -> None:
        worker = InferenceWorker(FakeEngine(), StreamingConfig())
        await worker.start()
        try:
            first = worker.submit(_frames())[1]
            second = worker.submit(_frames())[1]
        finally:
            await worker.stop()
        assert second == first + 1


class TestBacklogHandling:
    async def test_drops_the_oldest_when_the_queue_is_full(self) -> None:
        engine = FakeEngine()
        engine.gate.clear()  # hold the worker on its first job
        worker = InferenceWorker(engine, StreamingConfig(max_queued_jobs=2))
        await worker.start()
        try:
            worker.submit(_frames(1))
            await asyncio.sleep(0.05)  # let the worker pick up job 1

            assert worker.submit(_frames(2))[0] is SubmitOutcome.QUEUED
            assert worker.submit(_frames(3))[0] is SubmitOutcome.QUEUED
            # The queue is full; the next arrival must displace the oldest
            # waiting job rather than lengthen the backlog.
            outcome, _ = worker.submit(_frames(4))
            assert outcome is SubmitOutcome.REPLACED_OLDEST
            assert worker.queue_depth == 2
            assert worker.jobs_dropped == 1
        finally:
            engine.gate.set()
            await worker.stop()

    async def test_keeps_the_newest_work(self) -> None:
        engine = FakeEngine()
        engine.gate.clear()
        worker = InferenceWorker(engine, StreamingConfig(max_queued_jobs=1))
        await worker.start()
        try:
            worker.submit(_frames(1))
            await asyncio.sleep(0.05)
            worker.submit(_frames(2))
            worker.submit(_frames(3))
            engine.gate.set()
            for _ in range(2):
                await _drain_one(worker)
        finally:
            await worker.stop()
        # Segment 2 was displaced by the newer segment 3, so it never ran.
        assert engine.calls == [1, 3]

    async def test_discards_a_segment_that_waited_too_long(self) -> None:
        engine = FakeEngine()
        engine.gate.clear()
        worker = InferenceWorker(
            engine, StreamingConfig(max_queued_jobs=4, max_frame_age_seconds=0.1)
        )
        await worker.start()
        try:
            worker.submit(_frames(1))
            await asyncio.sleep(0.05)
            worker.submit(_frames(2))
            await asyncio.sleep(0.3)  # segment 2 goes stale while 1 is blocked
            engine.gate.set()
            first = await _drain_one(worker)
            second = await _drain_one(worker)
        finally:
            await worker.stop()

        assert first.dropped is False
        assert second.dropped is True
        assert second.result.text == ""
        assert "fell behind" in second.result.error
        assert engine.calls == [1]


class TestFailureIsolation:
    async def test_a_failing_segment_does_not_stop_the_worker(self) -> None:
        class Exploding(FakeEngine):
            def transcribe_frames(self, frames):
                if len(frames) == 1:
                    raise RuntimeError("CUDA is on fire")
                return super().transcribe_frames(frames)

        engine = Exploding()
        worker = InferenceWorker(engine, StreamingConfig())
        await worker.start()
        try:
            worker.submit(_frames(1))
            failed = await _drain_one(worker)
            worker.submit(_frames(2))
            recovered = await _drain_one(worker)
        finally:
            await worker.stop()

        assert failed.result.text == ""
        assert "RuntimeError" in failed.result.error
        assert recovered.result.text == "segment of 2"
        assert worker.is_running is False


class TestStats:
    async def test_counts_what_it_did(self) -> None:
        worker = InferenceWorker(FakeEngine(), StreamingConfig())
        await worker.start()
        try:
            worker.submit(_frames())
            await _drain_one(worker)
            stats = worker.stats()
        finally:
            await worker.stop()
        assert stats["submitted"] == 1
        assert stats["completed"] == 1
        assert stats["dropped"] == 0
