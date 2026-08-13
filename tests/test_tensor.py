"""Tensor construction: dimensions, dtype, cropping, grayscale and normalisation.

These checks matter because every one of these mistakes produces a model that
runs happily and returns confident nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.config import PreprocessingConfig
from ml.preprocessing.tensor import (
    LUMA_WEIGHTS,
    add_batch_dimension,
    denormalise_to_uint8,
    rois_to_tensor,
)


@pytest.fixture
def rois() -> np.ndarray:
    generator = np.random.default_rng(5)
    return generator.integers(0, 256, size=(12, 96, 96, 3), dtype=np.uint8)


class TestRoisToTensor:
    def test_output_shape_matches_the_model_contract(self, rois: np.ndarray) -> None:
        tensor = rois_to_tensor(rois)
        assert tensor.shape == (12, 1, 88, 88)

    def test_output_is_float32(self, rois: np.ndarray) -> None:
        assert rois_to_tensor(rois).dtype == torch.float32

    def test_accepts_grayscale_input(self) -> None:
        grey = np.random.default_rng(6).integers(0, 256, (8, 96, 96), dtype=np.uint8)
        assert rois_to_tensor(grey).shape == (8, 1, 88, 88)

    def test_rejects_a_float_array(self) -> None:
        with pytest.raises(TypeError):
            rois_to_tensor(np.zeros((4, 96, 96, 3), dtype=np.float32))

    def test_rejects_the_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError):
            rois_to_tensor(np.zeros((4, 96, 96, 4), dtype=np.uint8))

    def test_rejects_crops_smaller_than_the_network_input(self) -> None:
        with pytest.raises(ValueError, match="Cannot crop"):
            rois_to_tensor(np.zeros((4, 64, 64, 3), dtype=np.uint8))

    def test_normalisation_statistics_are_applied(self) -> None:
        """Grey maps to (luma - mean) / std.

        The BT.601 weights sum to 0.9999 rather than 1.0, and torchvision does
        not renormalise them. Matching that exactly matters: the checkpoint's
        input statistics were computed with the same slightly-off weights.
        """
        config = PreprocessingConfig()
        mid_grey = np.full((3, 96, 96, 3), 128, dtype=np.uint8)
        tensor = rois_to_tensor(mid_grey, config=config)
        luma = (128 / 255.0) * sum(LUMA_WEIGHTS)
        expected = (luma - config.pixel_mean) / config.pixel_std
        assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-5)

    def test_grayscale_uses_bt601_luma_weights(self) -> None:
        config = PreprocessingConfig()
        image = np.zeros((1, 96, 96, 3), dtype=np.uint8)
        image[..., 0] = 255  # pure red
        tensor = rois_to_tensor(image, config=config)
        expected = (LUMA_WEIGHTS[0] - config.pixel_mean) / config.pixel_std
        assert tensor.flatten()[0].item() == pytest.approx(expected, abs=1e-4)

    def test_centre_crop_keeps_the_middle(self) -> None:
        """A marker at the crop's centre must survive; a corner must not."""
        image = np.zeros((1, 96, 96, 3), dtype=np.uint8)
        image[0, 48, 48] = 255
        image[0, 0, 0] = 255
        tensor = rois_to_tensor(image)
        assert tensor[0, 0, 44, 44].item() > tensor[0, 0, 0, 0].item()

    def test_frame_order_is_preserved(self) -> None:
        frames = np.stack(
            [np.full((96, 96, 3), value, dtype=np.uint8) for value in (10, 100, 200)]
        )
        tensor = rois_to_tensor(frames)
        means = [tensor[index].mean().item() for index in range(3)]
        assert means[0] < means[1] < means[2]

    def test_does_not_mutate_the_input(self, rois: np.ndarray) -> None:
        original = rois.copy()
        rois_to_tensor(rois)
        np.testing.assert_array_equal(rois, original)


class TestBatchDimension:
    def test_adds_a_leading_axis(self) -> None:
        tensor = torch.zeros(10, 1, 88, 88)
        assert add_batch_dimension(tensor).shape == (1, 10, 1, 88, 88)

    def test_rejects_the_wrong_rank(self) -> None:
        with pytest.raises(ValueError):
            add_batch_dimension(torch.zeros(10, 88, 88))


class TestDenormalise:
    def test_round_trips_back_to_the_source_pixels(self) -> None:
        source = np.random.default_rng(9).integers(
            0, 256, (6, 96, 96), dtype=np.uint8
        )
        tensor = rois_to_tensor(source)
        recovered = denormalise_to_uint8(tensor)
        assert recovered.shape == (6, 88, 88)
        expected = source[:, 4:92, 4:92]
        assert np.abs(recovered.astype(int) - expected.astype(int)).max() <= 1

    def test_output_is_uint8(self) -> None:
        tensor = torch.randn(4, 1, 88, 88)
        assert denormalise_to_uint8(tensor).dtype == np.uint8
