"""Report whether the environment can run the pipeline.

Run this first when something does not work. It separates "a dependency is
missing", "the reference checkout is missing" and "the checkpoint is missing",
which produce very different-looking failures at runtime.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "mediapipe",
    "av",
    "sentencepiece",
    "safetensors",
    "fastapi",
    "uvicorn",
    "pydantic",
]
OPTIONAL = ["pytest", "httpx", "skimage", "pyarrow", "huggingface_hub", "requests"]


def _report_packages(names: list[str], label: str) -> list[str]:
    print(f"\n{label}")
    missing: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {name:<18} {version}")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            print(f"  {name:<18} MISSING ({type(exc).__name__})")
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")

    missing = _report_packages(REQUIRED, "Required packages")
    _report_packages(OPTIONAL, "Optional packages")

    print("\nCompute")
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            memory_gb = properties.total_memory / 1024**3
            print(f"  CUDA available: {properties.name} ({memory_gb:.1f} GB)")
            print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}")
        else:
            print(f"  CUDA not available; inference will run on CPU (torch {torch.__version__})")
    except ImportError:
        print("  torch not installed")

    print("\nReference assets")
    try:
        from ml.paths import (
            AUTO_AVSR_DIR,
            MEAN_FACE_PATH,
            SPM_MODEL_PATH,
            require_reference_assets,
        )

        require_reference_assets()
        print(f"  auto_avsr checkout: {AUTO_AVSR_DIR}")
        print(f"  mean face:          {MEAN_FACE_PATH.name}")
        print(f"  sentencepiece:      {SPM_MODEL_PATH.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {exc}")

    print("\nCheckpoints")
    try:
        from ml.models.checkpoints import CHECKPOINTS

        for key, info in CHECKPOINTS.items():
            if info.path.exists():
                size_gb = info.path.stat().st_size / 1e9
                print(f"  {key}: present ({size_gb:.2f} GB) at {info.path}")
            else:
                print(f"  {key}: MISSING - python scripts/download_checkpoint.py --name {key}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {exc}")

    if missing:
        print(f"\nMissing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return 1
    print("\nEnvironment looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
