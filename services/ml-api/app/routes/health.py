"""Liveness and model introspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas import HealthResponse, ModelSpecModel, ModelsResponse
from app.state import ServiceState, get_state
from ml.models.registry import available_models

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(state: ServiceState = Depends(get_state)) -> HealthResponse:
    """Report whether transcription is actually available.

    ``status`` stays ``ok`` when the model is missing: the process is healthy and
    can still preprocess. ``model_loaded`` is the field that decides whether the
    UI should offer transcription.
    """
    engine = state.engine
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_ready,
        device=state.device,
        model_name=engine.model.name,
        error=engine.load_error,
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def models(state: ServiceState = Depends(get_state)) -> ModelsResponse:
    """List registered adapters and the input contract of the active one."""
    spec = state.engine.model.spec
    return ModelsResponse(
        active=state.engine.model.name,
        spec=ModelSpecModel(**spec.to_dict()),
        available=available_models(),
    )
