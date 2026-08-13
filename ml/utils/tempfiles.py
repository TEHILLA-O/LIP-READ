"""Transient storage for uploads and captures.

Privacy default: nothing a user sends is kept. Media is written to a dedicated
temp directory, deleted in a ``finally`` block once inference returns, and any
survivors from a crashed process are swept on the next start. Persistence is
opt-in through :class:`~ml.config.PrivacyConfig`.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["TransientStore", "transient_path"]

_DIRECTORY_NAME = "vsr-transient"


class TransientStore:
    """Owns the directory that user media passes through."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / _DIRECTORY_NAME
        self.root.mkdir(parents=True, exist_ok=True)

    def new_path(self, suffix: str = ".mp4", prefix: str = "clip") -> Path:
        """Reserve a unique path without creating the file."""
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.root / f"{prefix}-{uuid.uuid4().hex}{safe_suffix}"

    @contextmanager
    def temporary_file(
        self, suffix: str = ".mp4", prefix: str = "clip", keep: bool = False
    ) -> Iterator[Path]:
        """Yield a path and delete it afterwards unless ``keep`` is set.

        Deletion runs even when the body raises, so a failed inference cannot
        leave a recording behind.
        """
        path = self.new_path(suffix=suffix, prefix=prefix)
        try:
            yield path
        finally:
            if keep:
                logger.warning("Retaining %s because persistence is enabled", path)
            else:
                self.remove(path)

    @staticmethod
    def remove(path: str | Path) -> bool:
        """Delete a file, tolerating a Windows lock from a reader still closing."""
        path = Path(path)
        for attempt in range(3):
            try:
                path.unlink(missing_ok=True)
                return True
            except PermissionError:
                if attempt == 2:
                    logger.error("Could not delete transient file %s", path)
                    return False
                time.sleep(0.05)
        return False

    def sweep(self, max_age_seconds: float = 900.0) -> int:
        """Delete leftovers older than ``max_age_seconds``. Returns the count."""
        cutoff = time.time() - max_age_seconds
        removed = 0
        for entry in self.root.glob("*"):
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    removed += self.remove(entry)
            except OSError:  # pragma: no cover - racing with another sweep
                continue
        if removed:
            logger.info("Swept %d stale transient file(s) from %s", removed, self.root)
        return removed

    def clear(self) -> int:
        """Delete everything in the store. Used on shutdown and in tests."""
        return self.sweep(max_age_seconds=-1.0)


_default_store: TransientStore | None = None


def default_store() -> TransientStore:
    global _default_store
    if _default_store is None:
        _default_store = TransientStore(os.environ.get("VSR_TEMP_DIR"))
    return _default_store


@contextmanager
def transient_path(suffix: str = ".mp4", keep: bool = False) -> Iterator[Path]:
    """Shorthand for a one-off transient file in the default store."""
    with default_store().temporary_file(suffix=suffix, keep=keep) as path:
        yield path
