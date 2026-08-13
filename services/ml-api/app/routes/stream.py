"""Live transcription over a WebSocket.

    browser -> frames -> rolling buffer -> segmenter -> queue -> worker
                                                                   |
    browser <----------------- transcripts <-----------------------+

The receive loop never runs inference. It decodes, detects and buffers, then
hands whole utterances to :class:`~ml.streaming.worker.InferenceWorker`, which
runs them on a thread. Results come back through a separate pump task, so a slow
transcription delays only the transcript, never the camera indicator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.live import LiveSession
from app.settings import get_settings
from app.state import get_state_ws
from ml.streaming.segmenter import SegmentEvent
from ml.streaming.worker import InferenceWorker, SubmitOutcome

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/v1/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    state = get_state_ws(websocket)
    settings = get_settings()

    session = LiveSession(state.config)
    worker = InferenceWorker(state.engine, state.config.streaming)
    await worker.start()
    pump = asyncio.create_task(_pump_results(websocket, worker))

    await _send(
        websocket,
        {
            "type": "ready",
            "model": state.engine.model.name,
            "model_loaded": state.engine.is_ready,
            "error": state.engine.load_error,
            "spec": state.engine.model.spec.to_dict(),
            "config": {
                "target_fps": state.config.preprocessing.target_fps,
                "min_utterance_seconds": state.config.streaming.min_utterance_seconds,
                "max_utterance_seconds": state.config.streaming.max_utterance_seconds,
                "buffer_seconds": state.config.streaming.buffer_seconds,
            },
        },
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                if len(data) > settings.max_websocket_message_bytes:
                    session.frames_rejected += 1
                    await _send(
                        websocket,
                        {"type": "error", "message": "Frame exceeds the size limit."},
                    )
                    continue
                await _handle_frame(websocket, session, worker, data, settings)
            elif (text := message.get("text")) is not None:
                if not await _handle_control(websocket, session, worker, text):
                    break
    except WebSocketDisconnect:
        logger.debug("Client disconnected from the live stream")
    except Exception:  # noqa: BLE001 - never leak a traceback onto the socket
        logger.exception("Live stream failed")
        await _send(websocket, {"type": "error", "message": "Internal stream error."})
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
        await worker.stop()
        session.close()
        if websocket.client_state is WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()


async def _handle_frame(
    websocket: WebSocket,
    session: LiveSession,
    worker: InferenceWorker,
    data: bytes,
    settings,
) -> None:
    # Decoding and face detection cost a few milliseconds each; on the event
    # loop they would stall every other connection's socket reads.
    image = await asyncio.to_thread(session.decode_frame, data)
    if image is None:
        session.frames_rejected += 1
        return

    observation = await asyncio.to_thread(session.observe, image)
    event = session.update_segmenter(observation)
    if event is not None:
        await _submit_segment(websocket, session, worker, event)

    if session.frames_received % settings.status_interval_frames == 0:
        await _send(
            websocket,
            {
                "type": "status",
                **observation.to_dict(),
                **session.stats(),
                "queue": worker.stats(),
            },
        )


async def _submit_segment(
    websocket: WebSocket,
    session: LiveSession,
    worker: InferenceWorker,
    event: SegmentEvent,
) -> None:
    frames = await asyncio.to_thread(session.collect_segment, event)
    outcome, sequence = worker.submit(frames)
    await _send(
        websocket,
        {
            "type": "segment",
            "sequence": sequence,
            "reason": event.reason.value,
            "duration": round(event.duration, 2),
            "frames": len(frames),
            "accepted": outcome is not SubmitOutcome.REJECTED,
            # Tell the UI plainly when the backlog forced work to be discarded,
            # rather than letting an utterance vanish without explanation.
            "displaced_earlier_segment": outcome is SubmitOutcome.REPLACED_OLDEST,
        },
    )


async def _handle_control(
    websocket: WebSocket, session: LiveSession, worker: InferenceWorker, text: str
) -> bool:
    """Process a JSON control message. Returns False to close the connection."""
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        await _send(websocket, {"type": "error", "message": "Malformed control message."})
        return True

    kind = message.get("type")
    if kind == "ping":
        await _send(websocket, {"type": "pong"})
    elif kind == "flush":
        # The speaker stopped or the tab is closing; transcribe what is open
        # instead of discarding a half-finished utterance.
        event = session.flush()
        if event is not None:
            await _submit_segment(websocket, session, worker, event)
        else:
            await _send(websocket, {"type": "segment", "sequence": -1, "accepted": False,
                                    "reason": "nothing_pending", "frames": 0,
                                    "duration": 0.0, "displaced_earlier_segment": False})
    elif kind == "reset":
        session.buffer.clear()
        session.segmenter.reset()
        session.motion.reset()
        session.tracker.reset()
        await _send(websocket, {"type": "reset"})
    elif kind == "close":
        return False
    else:
        await _send(websocket, {"type": "error", "message": f"Unknown command '{kind}'."})
    return True


async def _pump_results(websocket: WebSocket, worker: InferenceWorker) -> None:
    """Forward finished transcripts to the client as they complete."""
    async for job in worker.results():
        await _send(
            websocket,
            {
                "type": "transcript",
                "sequence": job.sequence,
                "queue_wait_ms": job.queue_wait_ms,
                "dropped": job.dropped,
                **job.result.to_dict(),
            },
        )


async def _send(websocket: WebSocket, payload: dict) -> None:
    """Send JSON, tolerating a client that has already gone away."""
    if websocket.client_state is not WebSocketState.CONNECTED:
        return
    try:
        await websocket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Dropped a message to a disconnected client")
