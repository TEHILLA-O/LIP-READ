"""Shared fixtures.

Test clips are generated on demand rather than committed, so the suite has no
binary fixtures to keep in sync and no real person's face in the repository.
Generation is session-scoped because encoding is the slow part.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# The service directory is hyphenated and so cannot be imported as a package
# path; put it on sys.path so the API tests can `import app.main` the same way
# uvicorn does when started from that directory.
sys.path.insert(0, str(PROJECT_ROOT / "services" / "ml-api"))

from ml.config import PreprocessingConfig  # noqa: E402
from ml.models.checkpoints import DEFAULT_CHECKPOINT, resolve_checkpoint  # noqa: E402
from ml.preprocessing.video_io import write_mp4  # noqa: E402
from scripts.make_test_videos import (  # noqa: E402
    _clip,
    _load_portrait,
    _noise_clip,
    _talking_clip,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs the 1 GB checkpoint or several seconds of compute"
    )
    config.addinivalue_line("markers", "model: needs the pretrained checkpoint")


@pytest.fixture(scope="session")
def portrait() -> np.ndarray:
    """A real face image, from scikit-image's bundled public-domain photograph."""
    return _load_portrait()


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("clips")


def _make_clip(
    directory: Path,
    name: str,
    frames: list[np.ndarray],
    fps: int,
) -> Path:
    path = directory / f"{name}.mp4"
    if not path.exists():
        write_mp4(frames, path, fps=fps)
    return path


@pytest.fixture(scope="session")
def still_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "still", _clip(30, "none", portrait), 25)


@pytest.fixture(scope="session")
def moving_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "moving", _clip(40, "translate", portrait), 25)


@pytest.fixture(scope="session")
def scaling_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "scaling", _clip(40, "scale", portrait), 25)


@pytest.fixture(scope="session")
def rotating_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "rotating", _clip(40, "rotate", portrait), 25)


@pytest.fixture(scope="session")
def clip_30fps(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "at30", _clip(60, "translate", portrait), 30)


@pytest.fixture(scope="session")
def clip_60fps(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "at60", _clip(120, "translate", portrait), 60)


@pytest.fixture(scope="session")
def clip_15fps(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "at15", _clip(30, "translate", portrait), 15)


@pytest.fixture(scope="session")
def two_faces_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(
        clip_dir, "two", _clip(40, "translate", portrait, second_face=True), 25
    )


@pytest.fixture(scope="session")
def no_face_clip(clip_dir: Path) -> Path:
    return _make_clip(clip_dir, "noface", _noise_clip(20), 25)


@pytest.fixture(scope="session")
def short_clip(clip_dir: Path, portrait: np.ndarray) -> Path:
    return _make_clip(clip_dir, "short", _clip(4, "none", portrait), 25)


@pytest.fixture(scope="session")
def face_frames(portrait: np.ndarray) -> list[np.ndarray]:
    """In-memory frames, for testing the live path without encoding."""
    return _clip(30, "translate", portrait)


@pytest.fixture(scope="session")
def talking_frames(portrait: np.ndarray) -> list[np.ndarray]:
    """Frames whose mouth moves, so utterance segmentation has something to find.

    Longer than ``max_utterance_seconds`` so the mouth never stops moving and the
    segmenter has to cut the utterance at the maximum window, which is the path
    continuous speech actually takes.
    """
    return _talking_clip(130, portrait)


@pytest.fixture(scope="session")
def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture(scope="session")
def checkpoint_available() -> bool:
    return resolve_checkpoint(DEFAULT_CHECKPOINT).path.exists()


@pytest.fixture
def requires_checkpoint(checkpoint_available: bool) -> None:
    if not checkpoint_available:
        pytest.skip("Checkpoint not downloaded; run scripts/download_checkpoint.py")
