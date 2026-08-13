"""Upload endpoints: transcribe a video, or just preprocess it.

Uploads are transient. The file is streamed to a temp path, used, and deleted in
a ``finally`` block, so a failed or cancelled request cannot leave a recording
on disk.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.schemas import PreprocessResponse, TranscriptionResponse
from app.settings import Settings, get_settings
from app.state import ServiceState, get_state
from ml.preprocessing.debug_export import export_mouth_track
from ml.preprocessing.pipeline import MouthROIPipeline, PreprocessingError
from ml.preprocessing.video_io import DecodeError
from ml.types import MouthTrack

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["transcribe"])

_DEBUG_KINDS = {"mouth_roi", "network_input"}
_CHUNK_SIZE = 1 << 20


async def _save_upload(
    upload: UploadFile, destination: Path, settings: Settings
) -> int:
    """Stream an upload to disk, enforcing the size cap as it goes.

    The declared Content-Length is not trusted; the limit is applied to bytes
    actually written.
    """
    written = 0
    with open(destination, "wb") as handle:
        while chunk := await upload.read(_CHUNK_SIZE):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"Upload exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit."
                    ),
                )
            handle.write(chunk)
    if written == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The file is empty.")
    return written


def _check_content_type(upload: UploadFile, settings: Settings) -> None:
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in settings.allowed_upload_types:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Accepted: {', '.join(sorted(settings.allowed_upload_types))}"
            ),
        )


def _suffix_for(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"} else ".mp4"


def _write_debug(state: ServiceState, track: MouthTrack) -> dict[str, str] | None:
    """Export debug MP4s into transient storage and return their URLs."""
    if not state.config.privacy.allow_debug_export:
        return None
    token = uuid.uuid4().hex
    export_mouth_track(
        track,
        state.store.root,
        stem=f"debug-{token}",
        config=state.config.preprocessing,
    )
    return {kind: f"/v1/debug/{token}/{kind}" for kind in sorted(_DEBUG_KINDS)}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="A video containing a speaking face"),
    debug: bool = Query(False, description="Also export the mouth ROI as MP4"),
    state: ServiceState = Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> TranscriptionResponse:
    """Transcribe speech from lip movement in an uploaded video."""
    _check_content_type(file, settings)

    path = state.store.new_path(suffix=_suffix_for(file), prefix="upload")
    keep = state.config.privacy.persist_uploads
    try:
        await _save_upload(file, path, settings)
        try:
            result, track = state.engine.transcribe_video(path, want_debug=debug)
        except DecodeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not read the video: {exc}",
            ) from exc

        payload = result.to_dict()
        if debug and track is not None:
            payload["debug"] = _write_debug(state, track)
        return TranscriptionResponse(**payload)
    finally:
        if not keep:
            state.store.remove(path)


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(
    file: UploadFile = File(...),
    debug: bool = Query(True, description="Export the mouth ROI as MP4"),
    state: ServiceState = Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> PreprocessResponse:
    """Run preprocessing only.

    Exists so the pipeline can be verified without loading a model, which makes
    it possible to debug framing and detection problems separately from
    transcription quality.
    """
    _check_content_type(file, settings)

    path = state.store.new_path(suffix=_suffix_for(file), prefix="upload")
    started = time.perf_counter()
    try:
        await _save_upload(file, path, settings)
        try:
            with MouthROIPipeline(state.config.preprocessing) as pipeline:
                track = pipeline.from_video(path)
        except PreprocessingError as exc:
            return PreprocessResponse(
                frames_processed=0,
                detected_frames=0,
                detection_ratio=0.0,
                mouth_detected=False,
                duration_seconds=0.0,
                source_fps=0.0,
                target_fps=state.config.preprocessing.target_fps,
                roi_shape=[],
                processing_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
        except DecodeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not read the video: {exc}",
            ) from exc

        return PreprocessResponse(
            frames_processed=track.total_frames,
            detected_frames=track.detected_frames,
            detection_ratio=round(track.detection_ratio, 4),
            mouth_detected=(
                track.detection_ratio
                >= state.config.preprocessing.min_detection_ratio
            ),
            duration_seconds=round(track.duration_seconds, 3),
            source_fps=round(track.source_fps, 2),
            target_fps=track.target_fps,
            roi_shape=list(track.rois.shape),
            processing_ms=int((time.perf_counter() - started) * 1000),
            debug=_write_debug(state, track) if debug else None,
        )
    finally:
        if not state.config.privacy.persist_uploads:
            state.store.remove(path)


@router.get("/debug/{token}/{kind}")
async def debug_artifact(
    token: str, kind: str, state: ServiceState = Depends(get_state)
) -> FileResponse:
    """Serve a debug MP4 written by a previous request.

    Artefacts live in transient storage and are swept on a timer, so a link
    stops working once the recording has been cleaned up.
    """
    if not state.config.privacy.allow_debug_export:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Debug export is disabled.")
    if kind not in _DEBUG_KINDS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown artefact '{kind}'.")
    if not token.isalnum():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Malformed token.")

    path = state.store.root / f"debug-{token}_{kind}.mp4"
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="That debug clip has expired and been deleted.",
        )
    return FileResponse(path, media_type="video/mp4", filename=f"{kind}.mp4")
