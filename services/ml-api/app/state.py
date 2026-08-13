"""Process-wide state: the model is loaded once, at start-up.

Loading a 250M-parameter model takes seconds and a gigabyte of GPU memory, so
doing it per request would be unusable. The engine is created and loaded during
the lifespan and shared by every request and WebSocket connection; it serialises
its own work internally.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fastapi import Request, WebSocket

from ml.config import AppConfig, resolve_device
from ml.inference.engine import InferenceEngine
from ml.utils.tempfiles import TransientStore

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    config: AppConfig
    engine: InferenceEngine
    store: TransientStore

    @property
    def device(self) -> str:
        return resolve_device(self.config.device)


def build_state() -> ServiceState:
    config = AppConfig.from_env()
    state = ServiceState(
        config=config,
        engine=InferenceEngine(config),
        store=TransientStore(os.environ.get("VSR_TEMP_DIR")),
    )
    # Anything left behind by a previous process was never meant to persist.
    swept = state.store.sweep(config.privacy.temp_sweep_age_seconds)
    if swept:
        logger.info("Swept %d stale transient file(s) at start-up", swept)
    return state


def get_state(request: Request) -> ServiceState:
    return request.app.state.service


def get_state_ws(websocket: WebSocket) -> ServiceState:
    return websocket.app.state.service


def get_engine(request: Request) -> InferenceEngine:
    return get_state(request).engine
