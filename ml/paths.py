"""Filesystem locations used across the ML package.

The Auto-AVSR research code (espnet subset), the mean-face reference array and
the SentencePiece vocabulary all live in the ``references/auto_avsr`` checkout
rather than being duplicated into this package. ``scripts/fetch_references.py``
creates that checkout. Every lookup goes through this module so there is a
single place to repoint if the checkout moves.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


REFERENCES_DIR: Path = _env_path("VSR_REFERENCES_DIR", PROJECT_ROOT / "references")
AUTO_AVSR_DIR: Path = REFERENCES_DIR / "auto_avsr"

#: Root that must be importable for ``import espnet.nets...`` to resolve.
AUTO_AVSR_IMPORT_ROOT: Path = AUTO_AVSR_DIR

#: 68-point mean face in a 256x256 frame; the alignment target for every clip.
MEAN_FACE_PATH: Path = (
    AUTO_AVSR_DIR / "preparation" / "detectors" / "mediapipe" / "20words_mean_face.npy"
)

#: SentencePiece unigram-5000 model and unit list the checkpoints were trained with.
SPM_MODEL_PATH: Path = AUTO_AVSR_DIR / "spm" / "unigram" / "unigram5000.model"
SPM_UNITS_PATH: Path = AUTO_AVSR_DIR / "spm" / "unigram" / "unigram5000_units.txt"

CHECKPOINTS_DIR: Path = _env_path("VSR_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints")
DEBUG_OUTPUT_DIR: Path = _env_path("VSR_DEBUG_DIR", PROJECT_ROOT / "debug_out")
TEST_VIDEO_DIR: Path = _env_path("VSR_TEST_VIDEO_DIR", PROJECT_ROOT / "data" / "test_videos")

_SETUP_HINT = (
    "Run `python scripts/fetch_references.py` to create the reference checkout."
)


class MissingReferenceError(RuntimeError):
    """Raised when required research code or data assets are not on disk."""


def require_reference_assets() -> None:
    """Fail loudly, and with a fix, if the reference checkout is incomplete.

    Called before any attempt to build a model so that a missing checkout is
    reported as a setup problem instead of an obscure ImportError.
    """
    missing = [
        str(path)
        for path in (
            AUTO_AVSR_IMPORT_ROOT / "espnet",
            MEAN_FACE_PATH,
            SPM_MODEL_PATH,
            SPM_UNITS_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise MissingReferenceError(
            "Missing Auto-AVSR reference assets:\n  "
            + "\n  ".join(missing)
            + f"\n{_SETUP_HINT}"
        )


def ensure_auto_avsr_importable() -> None:
    """Put the reference checkout on ``sys.path`` for its absolute imports.

    The vendored espnet subset imports itself as ``espnet.nets.*``, so its
    parent directory has to be importable. Kept idempotent because adapters may
    be constructed more than once per process.
    """
    import sys

    require_reference_assets()
    root = str(AUTO_AVSR_IMPORT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
