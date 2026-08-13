"""Mouth ROI sequence to network input tensor.

Mirrors Auto-AVSR's test-time ``VideoTransform``: scale to [0, 1], centre crop
96 -> 88, convert to grayscale with BT.601 luma weights, then normalise with the
dataset statistics the checkpoints were trained on. Getting any of these wrong
produces confident nonsense rather than an error, so the steps are kept together
and covered by tests.
"""

from __future__ import annotations

import numpy as np
import torch

from ml.config import PreprocessingConfig

__all__ = [
    "rois_to_tensor",
    "add_batch_dimension",
    "denormalise_to_uint8",
    "LUMA_WEIGHTS",
]

#: BT.601 luma weights, matching ``torchvision.transforms.Grayscale``.
LUMA_WEIGHTS: tuple[float, float, float] = (0.2989, 0.587, 0.114)


def _centre_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Centre crop the trailing two dimensions, matching torchvision's offsets."""
    height, width = tensor.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"Cannot crop {size}x{size} out of {height}x{width}")
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return tensor[..., top : top + size, left : left + size]


def rois_to_tensor(
    rois: np.ndarray,
    config: PreprocessingConfig | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert uint8 mouth crops to the model's input tensor.

    Args:
        rois: ``(T, H, W, 3)`` uint8 RGB, or ``(T, H, W)`` if already grayscale.
        config: Preprocessing settings; defaults to the Auto-AVSR contract.
        device: Where the resulting tensor should live.

    Returns:
        ``(T, 1, 88, 88)`` float32 tensor, normalised.
    """
    config = config or PreprocessingConfig()
    if rois.ndim == 3:
        rois = rois[..., None].repeat(3, axis=-1)
    if rois.ndim != 4 or rois.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) mouth crops, got {rois.shape}")
    if rois.dtype != np.uint8:
        raise TypeError(f"Expected uint8 mouth crops, got {rois.dtype}")

    # from_numpy shares memory; permute is a view, so this costs one copy total.
    frames = torch.from_numpy(np.ascontiguousarray(rois))
    frames = frames.to(device=device, non_blocking=True)
    frames = frames.permute(0, 3, 1, 2).float().div_(255.0)

    frames = _centre_crop(frames, config.network_input_size)

    red, green, blue = frames.unbind(dim=-3)
    grey = LUMA_WEIGHTS[0] * red + LUMA_WEIGHTS[1] * green + LUMA_WEIGHTS[2] * blue
    grey = grey.unsqueeze(dim=-3)

    return grey.sub_(config.pixel_mean).div_(config.pixel_std)


def add_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """``(T, 1, H, W)`` -> ``(1, T, 1, H, W)``, the front-end's expected layout."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected a (T, C, H, W) tensor, got {tensor.shape}")
    return tensor.unsqueeze(0)


def denormalise_to_uint8(
    tensor: torch.Tensor, config: PreprocessingConfig | None = None
) -> np.ndarray:
    """Invert normalisation so the exact network input can be watched as video."""
    config = config or PreprocessingConfig()
    frames = tensor.detach().to("cpu", torch.float32)
    if frames.ndim == 4:
        frames = frames.squeeze(1)
    frames = frames.mul(config.pixel_std).add(config.pixel_mean).clamp_(0.0, 1.0)
    return (frames * 255.0).round().to(torch.uint8).numpy()
