"""Measure transcription accuracy against LRS3 ground truth.

Two modes, which answer two different questions. Keeping them separate is the
only way to tell a modelling problem from a preprocessing problem.

``--source crops`` feeds the reference-preprocessed 96x96 mouth sequences
straight to the model, bypassing this project's detection and alignment
entirely. It asks: are the weights, tensor layout, vocabulary and beam search
wired up correctly? Expect roughly 20% WER, matching the published number. Near
0% would mean the label is leaking; above ~50% means the model path is wrong.

``--source pipeline`` runs whole video files through the real chain — decode,
detect, track, align, crop, tensorise, infer, decode. It asks: does our
preprocessing preserve what the model needs? Run it on the clips from
``scripts/make_talking_face_video.py``, which carry the same ground truth. WER
will be worse than in crop mode; the gap between the two is the cost of our
preprocessing, and a large gap is the signal to go and look at the debug MP4s.

Run ``python scripts/fetch_lrs3_samples.py`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import AppConfig  # noqa: E402
from ml.evaluation.metrics import corpus_wer, word_error_rate  # noqa: E402
from ml.inference.engine import InferenceEngine  # noqa: E402
from ml.models.registry import create_model  # noqa: E402
from ml.paths import TEST_VIDEO_DIR  # noqa: E402
from ml.types import MouthTrack  # noqa: E402

CROP_DIR = TEST_VIDEO_DIR / "lrs3"
COMPOSITE_DIR = TEST_VIDEO_DIR / "composite"


def _load_crop_samples() -> list[tuple[str, np.ndarray, str]]:
    manifest_path = CROP_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No samples at {CROP_DIR}. Run: python scripts/fetch_lrs3_samples.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    samples = []
    for entry in manifest:
        stem = entry["stem"]
        frames = np.load(CROP_DIR / f"{stem}.npz")["frames"]
        transcript = (CROP_DIR / f"{stem}.txt").read_text(encoding="utf-8").strip()
        samples.append((stem, frames, transcript))
    return samples


def _load_video_samples(directory: Path) -> list[tuple[str, Path, str]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No clips at {directory}. Run: python scripts/make_talking_face_video.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        (entry["file"], directory / entry["file"], entry["transcript"])
        for entry in manifest
    ]


def _as_track(frames: np.ndarray, fps: int) -> MouthTrack:
    """Wrap reference crops as a MouthTrack without re-running preprocessing."""
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    total = frames.shape[0]
    return MouthTrack(
        rois=frames.astype(np.uint8, copy=False),
        boxes=[None] * total,
        aligned_keypoints=np.zeros((total, 4, 2), dtype=np.float32),
        detected_frames=total,
        total_frames=total,
        source_fps=float(fps),
        target_fps=fps,
    )


def _report(name: str, reference: str, predicted: str, detail: str) -> tuple[str, str]:
    print(f"--- {name}  ({detail}) ---")
    print(f"  reference : {reference}")
    print(f"  predicted : {predicted}")
    print(f"  WER       : {word_error_rate(reference, predicted)}")
    return reference, predicted


def _evaluate_crops(config: AppConfig, device: str) -> list[tuple[str, str]]:
    samples = _load_crop_samples()
    print(f"Loaded {len(samples)} LRS3 crop sample(s) from {CROP_DIR}\n")

    model = create_model(
        "auto_avsr",
        device=device,
        preprocessing=config.preprocessing,
        decoding=config.decoding,
    )
    started = time.perf_counter()
    model.load()
    print(f"Model ready in {time.perf_counter() - started:.1f}s "
          f"(beam={config.decoding.beam_size}, device={model.device})\n")

    pairs: list[tuple[str, str]] = []
    for stem, frames, reference in samples:
        track = _as_track(frames, config.preprocessing.target_fps)
        started = time.perf_counter()
        hypotheses, raw = model.transcribe(track)
        elapsed_ms = (time.perf_counter() - started) * 1000

        best = hypotheses[0]
        pairs.append(_report(
            stem, reference, best.text,
            f"{track.total_frames} frames, {elapsed_ms:.0f} ms",
        ))
        print(f"  confidence: {best.confidence:.3f}   encoded {raw.frames_encoded} frames")
        for alternative in hypotheses[1:3]:
            print(f"  alt       : {alternative.text}  ({alternative.confidence:.3f})")
        print()

    model.unload()
    return pairs


def _evaluate_pipeline(config: AppConfig, directory: Path) -> list[tuple[str, str]]:
    samples = _load_video_samples(directory)
    print(f"Loaded {len(samples)} composite clip(s) from {directory}\n")

    engine = InferenceEngine(config)
    started = time.perf_counter()
    engine.load()
    if not engine.is_ready:
        raise SystemExit(f"Model unavailable: {engine.load_error}")
    print(f"Model ready in {time.perf_counter() - started:.1f}s "
          f"(beam={config.decoding.beam_size}, device={config.device})\n")

    pairs: list[tuple[str, str]] = []
    try:
        for name, path, reference in samples:
            result, _ = engine.transcribe_video(path)
            if result.error:
                print(f"--- {name} ---\n  no transcript: {result.error}\n")
                pairs.append((reference, ""))
                continue

            pairs.append(_report(
                name, reference, result.text,
                f"{result.frames_processed} frames, {result.processing_ms} ms",
            ))
            print(f"  confidence: {result.confidence:.3f}   "
                  f"detection {result.diagnostics['detection_ratio']:.0%}")
            for alternative in result.alternatives[:2]:
                print(f"  alt       : {alternative.text}  ({alternative.confidence:.3f})")
            print()
    finally:
        engine.close()
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["crops", "pipeline"], default="crops",
        help="Reference crops (model only) or full videos (whole pipeline)",
    )
    parser.add_argument("--clips", type=Path, default=COMPOSITE_DIR)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    config = AppConfig.from_env().with_device(arguments.device)
    config = replace(
        config, decoding=replace(config.decoding, beam_size=arguments.beam_size)
    )

    if arguments.source == "crops":
        pairs = _evaluate_crops(config, arguments.device)
    else:
        pairs = _evaluate_pipeline(config, arguments.clips)

    print(f"Corpus WER over {len(pairs)} utterances: {corpus_wer(pairs)}")
    print("Published Auto-AVSR WER on the full LRS3 test set: 20.3%")
    if arguments.source == "pipeline":
        print(
            "Composite clips graft one person's mouth onto another's face and "
            "survive two resampling round-trips, so they read harder than real "
            "video. Compare against --source crops on the same samples."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
