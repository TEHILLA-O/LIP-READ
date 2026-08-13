"""Pretrained checkpoint catalogue, download and verification.

Weights are never bundled and never fetched implicitly during a request. If a
checkpoint is missing, inference fails with instructions instead of falling back
to an untrained network, which would emit fluent, entirely invented text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml.models.base import CheckpointMissingError
from ml.paths import CHECKPOINTS_DIR

__all__ = [
    "CheckpointInfo",
    "CHECKPOINTS",
    "DEFAULT_CHECKPOINT",
    "resolve_checkpoint",
    "checkpoint_path",
    "download_checkpoint",
    "sha256_of",
]


@dataclass(frozen=True)
class CheckpointInfo:
    """A downloadable set of weights."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    #: Word error rate reported on LRS3, for choosing between variants.
    lrs3_wer: float
    description: str
    subdirectory: str = "auto_avsr"
    license_note: str = ""

    @property
    def path(self) -> Path:
        return CHECKPOINTS_DIR / self.subdirectory / self.filename


#: Auto-AVSR visual checkpoint, mirrored on the Hugging Face Hub. The upstream
#: repository distributes it through Google Drive, which is not resumable and
#: rate-limits automated downloads; the mirror carries identical weights.
CHECKPOINTS: dict[str, CheckpointInfo] = {
    "auto_avsr_vsr_base": CheckpointInfo(
        key="auto_avsr_vsr_base",
        filename="vsr_trlrwlrs2lrs3vox2avsp_base.safetensors",
        url=(
            "https://huggingface.co/nguyenvulebinh/"
            "auto_avsr_visual_trlrwlrs2lrs3vox2avsp_base/resolve/main/model.safetensors"
        ),
        sha256="2fa62ba8750af78d45c5dcddc4eb48bb2187cc12a850903171c45302a8422981",
        size_bytes=1_001_736_152,
        lrs3_wer=20.3,
        description=(
            "Auto-AVSR visual-only Conformer, trained on LRW + LRS2 + LRS3 + "
            "VoxCeleb2 + AVSpeech (3,291 h). 250M parameters."
        ),
        license_note=(
            "Code is Apache 2.0. The weights inherit the terms of the datasets "
            "they were trained on and are intended for research use."
        ),
    ),
}

DEFAULT_CHECKPOINT = "auto_avsr_vsr_base"


def resolve_checkpoint(key: str = DEFAULT_CHECKPOINT) -> CheckpointInfo:
    if key not in CHECKPOINTS:
        raise KeyError(f"Unknown checkpoint '{key}'. Known: {sorted(CHECKPOINTS)}")
    return CHECKPOINTS[key]


def checkpoint_path(key: str = DEFAULT_CHECKPOINT, *, verify: bool = False) -> Path:
    """Return the local path of a checkpoint, or explain how to get it."""
    info = resolve_checkpoint(key)
    if not info.path.exists():
        raise CheckpointMissingError(
            f"Checkpoint '{key}' is not present at {info.path}.\n"
            f"{info.description}\n"
            f"Download it with: python scripts/download_checkpoint.py --name {key}\n"
            f"({info.size_bytes / 1e9:.1f} GB from {info.url})"
        )
    if verify:
        actual = sha256_of(info.path)
        if actual != info.sha256:
            raise CheckpointMissingError(
                f"Checkpoint '{key}' at {info.path} is corrupt.\n"
                f"  expected sha256 {info.sha256}\n  actual   sha256 {actual}\n"
                "Delete it and download again."
            )
    return info.path


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    key: str = DEFAULT_CHECKPOINT,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch a checkpoint and verify its hash.

    Downloads to a ``.part`` file and renames on success, so an interrupted
    download can never be mistaken for a complete one.
    """
    import requests

    info = resolve_checkpoint(key)
    destination = info.path
    if destination.exists() and not force:
        if sha256_of(destination) == info.sha256:
            return destination
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    with requests.get(info.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", info.size_bytes))
        written = 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total)

    actual = sha256_of(partial)
    if actual != info.sha256:
        partial.unlink(missing_ok=True)
        raise CheckpointMissingError(
            f"Downloaded checkpoint hash mismatch for '{key}'.\n"
            f"  expected {info.sha256}\n  actual   {actual}"
        )

    partial.replace(destination)
    return destination
