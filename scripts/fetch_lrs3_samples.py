"""Fetch a few ground-truth samples from the LRS3 test set.

LRS3 is the benchmark Auto-AVSR reports 20.3% WER on, and this mirror stores the
already-cropped 96x96 mouth sequences alongside their reference transcripts.
That makes it the one dataset that can answer "is the model wired up correctly?"
separately from "is my preprocessing correct?" — feed it the reference crops and
the transcript should come out close to the label.

Only the parquet footer and the row groups actually needed are transferred, so
this costs far less than the 346 MB shard.

Samples are written to ``data/test_videos/lrs3/`` as:
    sample_NN.npz  exact uint8 mouth frames, for numerical checks
    sample_NN.mp4  the same frames, for watching
    sample_NN.txt  the reference transcript
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import TEST_VIDEO_DIR  # noqa: E402
from ml.preprocessing.video_io import write_mp4  # noqa: E402

DATASET = "mattymchen/lrs3-test"
SHARD = "data/train-00000-of-00002-6d235766a16b101a.parquet"
OUTPUT_DIR = TEST_VIDEO_DIR / "lrs3"


def _open_parquet():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    handle = filesystem.open(f"datasets/{DATASET}/{SHARD}", "rb")
    return pq.ParquetFile(handle)


def describe() -> None:
    parquet = _open_parquet()
    metadata = parquet.metadata
    print(f"rows={metadata.num_rows}  row_groups={metadata.num_row_groups}")
    print(f"columns={parquet.schema_arrow.names}")
    group = metadata.row_group(0)
    print(f"row group 0: {group.num_rows} rows, {group.total_byte_size / 1e6:.1f} MB")


def _to_frames(raw) -> np.ndarray:
    """Nested list column -> ``(T, H, W)`` uint8."""
    frames = np.array(raw.as_py() if hasattr(raw, "as_py") else raw, dtype=np.uint8)
    if frames.ndim != 3:
        raise ValueError(f"Expected (T, H, W) frames, got shape {frames.shape}")
    return frames


def fetch(count: int) -> list[dict[str, object]]:
    parquet = _open_parquet()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batches = parquet.iter_batches(batch_size=count, columns=["idx", "video", "label"])
    batch = next(iter(batches))

    manifest: list[dict[str, object]] = []
    for position in range(min(count, batch.num_rows)):
        frames = _to_frames(batch.column("video")[position])
        label = str(batch.column("label")[position])
        stem = f"sample_{position:02d}"

        np.savez_compressed(OUTPUT_DIR / f"{stem}.npz", frames=frames)
        write_mp4(frames, OUTPUT_DIR / f"{stem}.mp4", fps=25)
        (OUTPUT_DIR / f"{stem}.txt").write_text(label, encoding="utf-8")

        entry = {
            "stem": stem,
            "frames": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "seconds": round(frames.shape[0] / 25, 2),
            "transcript": label,
        }
        manifest.append(entry)
        print(f"{stem}: {entry['frames']} frames {entry['width']}x{entry['height']}  {label!r}")

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(manifest)} sample(s) to {OUTPUT_DIR}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4, help="Samples to download")
    parser.add_argument("--describe", action="store_true", help="Only print metadata")
    arguments = parser.parse_args()

    if arguments.describe:
        describe()
        return 0
    fetch(arguments.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
