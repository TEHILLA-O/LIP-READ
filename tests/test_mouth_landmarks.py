"""Lip geometry measurement.

The live segmenter trusts these measurements to be independent of where the head
is and how big it appears, so the invariances are tested directly rather than
only through the segmenter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.mouth_landmarks import INNER_LIP_CONTOUR, MouthLandmarker
from scripts.make_test_videos import _background, _paste_face, _talking_clip


@pytest.fixture(scope="module")
def landmarker() -> MouthLandmarker:
    with MouthLandmarker() as instance:
        yield instance


def _framed(portrait: np.ndarray, centre: tuple[int, int], scale: float) -> np.ndarray:
    frame = _background(0)
    _paste_face(frame, portrait, centre, scale)
    return frame


class TestMeasurement:
    def test_measures_a_face(self, landmarker: MouthLandmarker, portrait: np.ndarray) -> None:
        shape = landmarker.measure(_framed(portrait, (320, 240), 0.62))
        assert shape is not None
        assert shape.aperture > 0.0
        assert shape.width > 0.0
        assert shape.interocular > 1.0
        assert shape.contour.shape == (len(INNER_LIP_CONTOUR), 2)

    def test_returns_none_without_a_face(self, landmarker: MouthLandmarker) -> None:
        assert landmarker.measure(np.zeros((240, 320, 3), dtype=np.uint8)) is None

    def test_mouth_box_contains_the_lip_contour(
        self, landmarker: MouthLandmarker, portrait: np.ndarray
    ) -> None:
        shape = landmarker.measure(_framed(portrait, (320, 240), 0.62))
        box = shape.box
        assert box.width > 0 and box.height > 0
        assert (shape.contour[:, 0] >= box.x).all()
        assert (shape.contour[:, 0] <= box.x + box.width).all()
        assert (shape.contour[:, 1] >= box.y).all()
        assert (shape.contour[:, 1] <= box.y + box.height).all()

    def test_rejects_non_uint8_frames(self, landmarker: MouthLandmarker) -> None:
        with pytest.raises(TypeError):
            landmarker.measure(np.zeros((240, 320, 3), dtype=np.float32))

    def test_rejects_grayscale_frames(self, landmarker: MouthLandmarker) -> None:
        with pytest.raises(ValueError):
            landmarker.measure(np.zeros((240, 320), dtype=np.uint8))


class TestNormalisation:
    """Aperture must be a shape, not a pixel count.

    Invariance to head pose is what makes the signal usable, but it cannot be
    measured numerically on this fixture: the portrait's face spans only about
    45 pixels between the eyes, and FaceMesh's fit on it shifts noticeably when
    the face is moved or resized. The property is therefore checked two ways
    that are stable — the arithmetic here, and the behaviour in
    ``test_live_session.py``, where a translating head must not open an
    utterance.
    """

    def test_aperture_is_the_inner_lip_gap_over_inter_ocular_distance(
        self, landmarker: MouthLandmarker, portrait: np.ndarray
    ) -> None:
        shape = landmarker.measure(_framed(portrait, (320, 240), 0.62))
        assert shape is not None
        upper = shape.contour[INNER_LIP_CONTOUR.index(13)]
        lower = shape.contour[INNER_LIP_CONTOUR.index(14)]
        gap = float(np.linalg.norm(upper - lower))
        assert shape.aperture * shape.interocular == pytest.approx(gap, rel=1e-4)
        # A pixel gap this small would be a plausible aperture if unnormalised,
        # so confirm the two really are on different scales.
        assert gap > 1.0
        assert shape.aperture < 1.0

    def test_centre_is_the_middle_of_the_lip_gap(
        self, landmarker: MouthLandmarker, portrait: np.ndarray
    ) -> None:
        shape = landmarker.measure(_framed(portrait, (320, 240), 0.62))
        assert shape is not None
        upper = shape.contour[INNER_LIP_CONTOUR.index(13)]
        lower = shape.contour[INNER_LIP_CONTOUR.index(14)]
        assert shape.centre == pytest.approx((upper + lower) / 2.0, rel=1e-4)


class TestResponse:
    def test_aperture_follows_the_mouth_opening(
        self, landmarker: MouthLandmarker, portrait: np.ndarray
    ) -> None:
        apertures = [
            shape.aperture
            for shape in (landmarker.measure(f) for f in _talking_clip(40, portrait))
            if shape is not None
        ]
        assert len(apertures) > 30
        # An opening and closing mouth must move the measurement substantially,
        # or the segmenter has nothing to key on.
        assert max(apertures) > 2 * min(apertures)
