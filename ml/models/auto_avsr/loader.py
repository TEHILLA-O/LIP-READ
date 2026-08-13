"""Builds the Auto-AVSR E2E network and loads pretrained weights.

The published research code and the mirrored checkpoint disagree on parameter
names: the checkpoint keeps the visual front-end inside the encoder
(``vsr.encoder.frontend.*``, ``vsr.encoder.embed.0.*``) while ``E2E`` hoists them
out (``frontend.*``, ``proj_encoder.*``). The rename is mechanical but must be
exact — a silently partial load yields a model that runs and returns fluent
nonsense, so every key is accounted for here.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ml.models.base import CheckpointMissingError
from ml.paths import ensure_auto_avsr_importable

__all__ = ["build_e2e", "load_state_dict_into", "remap_checkpoint_keys", "read_state_dict"]

#: Applied longest-prefix first; the first match wins.
_KEY_REWRITES: tuple[tuple[str, str], ...] = (
    ("vsr.encoder.frontend.", "frontend."),
    ("vsr.encoder.embed.0.", "proj_encoder."),
    ("vsr.encoder.", "encoder."),
    ("vsr.decoder.", "decoder."),
    ("vsr.ctc.", "ctc."),
)

#: Keys the E2E module owns but no inference path uses, so their absence from a
#: checkpoint is harmless.
_IGNORABLE_MISSING_PREFIXES = ("criterion.",)


def build_e2e(odim: int, modality: str = "video") -> torch.nn.Module:
    """Instantiate the Conformer encoder / Transformer decoder network."""
    ensure_auto_avsr_importable()
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

    return E2E(odim, modality)


def read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load weights from ``.safetensors`` or ``.pth``."""
    path = Path(path)
    if not path.exists():
        raise CheckpointMissingError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def remap_checkpoint_keys(
    state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Rename mirrored-checkpoint keys onto the ``E2E`` module layout.

    Checkpoints already using the research layout pass through untouched.
    """
    if not any(key.startswith("vsr.") for key in state_dict):
        return state_dict

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for old, new in _KEY_REWRITES:
            if key.startswith(old):
                remapped[new + key[len(old) :]] = value
                break
        else:
            remapped[key] = value
    return remapped


def load_state_dict_into(model: torch.nn.Module, path: str | Path) -> dict[str, int]:
    """Load a checkpoint and verify nothing important was skipped.

    Returns a small report for logging. Raises if any weight the network needs
    was absent, rather than leaving it randomly initialised.
    """
    state_dict = remap_checkpoint_keys(read_state_dict(path))
    result = model.load_state_dict(state_dict, strict=False)

    missing = [
        key
        for key in result.missing_keys
        if not key.startswith(_IGNORABLE_MISSING_PREFIXES)
    ]
    if missing:
        preview = ", ".join(missing[:8])
        raise CheckpointMissingError(
            f"Checkpoint {path} is missing {len(missing)} weights required by the "
            f"model, for example: {preview}. Refusing to run a partially "
            "initialised network."
        )

    return {
        "loaded": len(state_dict) - len(result.unexpected_keys),
        "unexpected": len(result.unexpected_keys),
        "ignored_missing": len(result.missing_keys) - len(missing),
    }
