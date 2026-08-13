"""Download and verify a pretrained checkpoint.

Weights are large and are never fetched implicitly at request time, so this is
the one place that pulls them. The hash is checked before the file is put in
place: a truncated download that loads without complaint would produce fluent,
entirely invented transcripts.

    python scripts/download_checkpoint.py
    python scripts/download_checkpoint.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.base import CheckpointMissingError  # noqa: E402
from ml.models.checkpoints import (  # noqa: E402
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    download_checkpoint,
    resolve_checkpoint,
    sha256_of,
)


def _list() -> None:
    for key, info in CHECKPOINTS.items():
        present = "present" if info.path.exists() else "missing"
        print(f"{key}  [{present}]")
        print(f"  {info.description}")
        print(f"  {info.size_bytes / 1e9:.1f} GB, LRS3 WER {info.lrs3_wer}%")
        print(f"  {info.path}")
        if info.license_note:
            print(f"  {info.license_note}")
        print()


class _Progress:
    """Single-line progress, printed at most a few times a second."""

    def __init__(self) -> None:
        self._last = 0.0

    def __call__(self, written: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last < 0.25 and written < total:
            return
        self._last = now
        share = written / total if total else 0
        bar = "#" * int(share * 30)
        print(
            f"\r  [{bar:<30}] {written / 1e9:.2f}/{total / 1e9:.2f} GB",
            end="",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_CHECKPOINT, help="Checkpoint key")
    parser.add_argument("--list", action="store_true", help="Show the catalogue")
    parser.add_argument("--force", action="store_true", help="Re-download if present")
    parser.add_argument("--verify", action="store_true", help="Only check the hash")
    arguments = parser.parse_args()

    if arguments.list:
        _list()
        return 0

    try:
        info = resolve_checkpoint(arguments.name)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    if arguments.verify:
        if not info.path.exists():
            print(f"Not downloaded: {info.path}", file=sys.stderr)
            return 1
        actual = sha256_of(info.path)
        ok = actual == info.sha256
        print(f"{'OK' if ok else 'CORRUPT'}  {info.path}\n  sha256 {actual}")
        return 0 if ok else 1

    print(f"{info.key}: {info.description}")
    if info.license_note:
        print(f"  {info.license_note}")
    print(f"  from {info.url}")

    try:
        path = download_checkpoint(
            arguments.name, force=arguments.force, progress=_Progress()
        )
    except CheckpointMissingError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report the cause, not a traceback
        print(f"\nDownload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\nVerified and saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
