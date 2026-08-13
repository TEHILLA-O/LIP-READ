"""Transcribe a video from the command line.

The Phase 1 entry point: runs the whole chain (decode, detect, track, align,
crop, tensorise, infer, decode) and can write the debug MP4s that show what the
model was actually given.

    python scripts/transcribe.py data/test_videos/synthetic/face_still.mp4 --debug
    python scripts/transcribe.py clip.mp4 --preprocess-only --debug

``--preprocess-only`` skips the model entirely, so preprocessing can be checked
before a checkpoint exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import AppConfig  # noqa: E402
from ml.inference.engine import InferenceEngine  # noqa: E402
from ml.paths import DEBUG_OUTPUT_DIR  # noqa: E402
from ml.preprocessing.debug_export import (  # noqa: E402
    export_mouth_track,
    export_source_overlay,
)
from ml.preprocessing.pipeline import MouthROIPipeline, PreprocessingError  # noqa: E402
from ml.preprocessing.video_io import probe  # noqa: E402


def _print_track(track, elapsed_ms: float) -> None:
    print(f"  frames extracted : {track.total_frames}")
    print(f"  faces detected   : {track.detected_frames} "
          f"({track.detection_ratio:.0%})")
    print(f"  roi array        : {track.rois.shape} {track.rois.dtype}")
    print(f"  duration         : {track.duration_seconds:.2f}s "
          f"@ {track.target_fps} fps (source {track.source_fps:.2f} fps)")
    print(f"  preprocessing    : {elapsed_ms:.0f} ms")


def _write_debug(source: Path, track, stem: str, overlay: bool) -> None:
    export = export_mouth_track(track, DEBUG_OUTPUT_DIR, stem=stem)
    print(f"  debug mouth ROI  : {export.mouth_roi}")
    print(f"  debug net input  : {export.network_input}")
    if overlay:
        path = export_source_overlay(
            source, track, DEBUG_OUTPUT_DIR / f"{stem}_overlay.mp4"
        )
        print(f"  debug overlay    : {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--debug", action="store_true", help="Write debug MP4s")
    parser.add_argument("--overlay", action="store_true", help="Also write a source overlay")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print the result object")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not arguments.video.exists():
        print(f"No such file: {arguments.video}", file=sys.stderr)
        return 2

    config = AppConfig.from_env().with_device(arguments.device)
    if arguments.beam_size:
        from dataclasses import replace

        config = replace(
            config, decoding=replace(config.decoding, beam_size=arguments.beam_size)
        )

    metadata = probe(arguments.video)
    print(f"\n{arguments.video}")
    print(f"  source           : {metadata.width}x{metadata.height} "
          f"{metadata.codec} {metadata.fps:.2f} fps "
          f"{metadata.duration_seconds:.2f}s rotation={metadata.rotation}")

    stem = arguments.video.stem

    if arguments.preprocess_only:
        started = time.perf_counter()
        with MouthROIPipeline(config.preprocessing) as pipeline:
            try:
                track = pipeline.from_video(arguments.video)
            except PreprocessingError as exc:
                print(f"  preprocessing failed: {exc}")
                return 1
            elapsed_ms = (time.perf_counter() - started) * 1000
            _print_track(track, elapsed_ms)
            if arguments.debug:
                _write_debug(arguments.video, track, stem, arguments.overlay)
        return 0

    engine = InferenceEngine(config)
    engine.load()
    if not engine.is_ready:
        print(f"  model unavailable: {engine.load_error}")
        return 1

    try:
        result, track = engine.transcribe_video(arguments.video, want_debug=arguments.debug)
        if track is not None:
            _print_track(track, result.diagnostics.get("inference_ms", 0))
            if arguments.debug:
                _write_debug(arguments.video, track, stem, arguments.overlay)

        print()
        if result.error:
            print(f"  no transcript    : {result.error}")
        else:
            print(f"  transcript       : {result.text}")
            print(f"  confidence       : {result.confidence:.1%}")
            for alternative in result.alternatives:
                print(f"  alternative      : {alternative.text} "
                      f"({alternative.confidence:.1%})")
        print(f"  frames processed : {result.frames_processed}")
        print(f"  total time       : {result.processing_ms} ms")

        if arguments.json:
            print("\n" + json.dumps(result.to_dict(), indent=2))
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
