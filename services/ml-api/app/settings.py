"""Service settings, overridable with ``VSR_API_*`` environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VSR_API_", env_file=".env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8000

    #: The Next.js dev server. Widen only if the UI is served elsewhere.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    #: Upload ceiling. Enforced while streaming to disk, because a client can
    #: send any Content-Length it likes.
    max_upload_bytes: int = 200 * 1024 * 1024

    allowed_upload_types: list[str] = Field(
        default_factory=lambda: [
            "video/mp4",
            "video/quicktime",
            "video/webm",
            "video/x-matroska",
            "video/x-msvideo",
            "application/octet-stream",
        ]
    )

    #: Largest single WebSocket frame accepted, guarding against memory abuse.
    max_websocket_message_bytes: int = 4 * 1024 * 1024

    #: How often the live endpoint reports buffer and detection state.
    status_interval_frames: int = 5

    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
