"""Application entry point.

    uvicorn app.main:app --reload    (run from services/ml-api)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, stream, transcribe
from app.settings import get_settings
from app.state import build_state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    state = build_state()
    app.state.service = state

    # Weights are loaded once here rather than per request. It blocks start-up
    # for a few seconds, which is the right trade: the alternative is every
    # first request paying for it.
    logger.info("Loading '%s' on %s", state.config.model_name, state.device)
    await asyncio.to_thread(state.engine.load)
    if state.engine.is_ready:
        logger.info("Model ready")
    else:
        logger.warning(
            "Transcription unavailable: %s. Preprocessing endpoints still work.",
            state.engine.load_error,
        )

    sweeper = asyncio.create_task(_sweep_periodically(state), name="vsr-temp-sweeper")
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        state.engine.close()
        if not state.config.privacy.persist_uploads:
            state.store.clear()


async def _sweep_periodically(state) -> None:
    """Delete transient media a crashed or abandoned request left behind.

    The request path already deletes in a ``finally``; this is the backstop that
    keeps the privacy promise when the process dies mid-request.
    """
    interval = max(60.0, state.config.privacy.temp_sweep_age_seconds / 3)
    while True:
        await asyncio.sleep(interval)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                state.store.sweep, state.config.privacy.temp_sweep_age_seconds
            )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Visual Speech Recognition API",
        description=(
            "Transcribes speech from lip movement in video. Results are model "
            "predictions with a confidence score, never a guaranteed transcript."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(transcribe.router)
    app.include_router(stream.router)
    return app


app = create_app()
