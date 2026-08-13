"""Per-connection live session logic, tested without a WebSocket."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from app.live import LiveSession, _resample

from ml.config import AppConfig, PreprocessingConfig, StreamingConfig
from ml.streaming.buffer import BufferedFrame
from ml.streaming.segmenter import SegmenterState, SegmentReason


@pytest.fixture
def session() -> LiveSession:
    live = LiveSession(AppConfig(preprocessing=PreprocessingConfig()))
    yield live
    live.close()


def _buffered(timestamps: list[float]) -> list[BufferedFrame]:
    return [
        BufferedFrame(image=np.full((4, 4, 3), index, dtype=np.uint8),
                      timestamp=timestamp, index=index)
        for index, timestamp in enumerate(timestamps)
    ]


class TestFrameDecoding:
    def test_decodes_a_jpeg_to_rgb(self, portrait: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
        assert ok
        decoded = LiveSession.decode_frame(encoded.tobytes())
        assert decoded is not None
        assert decoded.shape == portrait.shape
        assert decoded.dtype == np.uint8
        # JPEG is lossy, so compare loosely; a BGR/RGB swap would be far worse.
        assert np.abs(decoded.astype(int) - portrait.astype(int)).mean() < 12

    def test_returns_none_for_rubbish(self) -> None:
        assert LiveSession.decode_frame(b"definitely not an image") is None


class TestObserve:
    def test_detects_the_mouth_and_buffers_the_frame(
        self, session: LiveSession, face_frames: list[np.ndarray]
    ) -> None:
        observation = session.observe(face_frames[0])
        assert observation.mouth_detected is True
        assert observation.face_box is not None
        assert observation.mouth_box.width > 0
        assert len(session.buffer) == 1

    def test_mouth_box_sits_inside_the_face_box(
        self, session: LiveSession, face_frames: list[np.ndarray]
    ) -> None:
        observation = session.observe(face_frames[0])
        face, mouth = observation.face_box, observation.mouth_box
        centre_x = mouth.x + mouth.width / 2
        centre_y = mouth.y + mouth.height / 2
        assert face.x <= centre_x <= face.x + face.width
        assert face.y <= centre_y <= face.y + face.height
        # The mouth is in the lower half of a face.
        assert centre_y > face.y + face.height / 2

    def test_reports_no_mouth_on_a_blank_frame(self, session: LiveSession) -> None:
        observation = session.observe(np.zeros((240, 320, 3), dtype=np.uint8))
        assert observation.mouth_detected is False
        assert observation.energy == 0.0
        assert observation.face_box is None

    def test_still_buffers_frames_without_a_face(self, session: LiveSession) -> None:
        # The buffer must stay aligned with real time, or a segment's frames
        # would no longer correspond to the window the segmenter measured.
        session.observe(np.zeros((240, 320, 3), dtype=np.uint8))
        assert len(session.buffer) == 1

    def test_tracks_the_arrival_rate(
        self, session: LiveSession, face_frames: list[np.ndarray]
    ) -> None:
        for index, frame in enumerate(face_frames[:6]):
            session.observe(frame, timestamp=index * 0.04)
        assert session.observed_fps == pytest.approx(25.0, abs=1.0)


class TestSegmentCollection:
    def test_collects_frames_for_a_closed_utterance(self, portrait: np.ndarray) -> None:
        config = AppConfig(streaming=StreamingConfig(min_utterance_seconds=0.2))
        session = LiveSession(config)
        try:
            for index in range(50):
                session.observe(portrait, timestamp=index * 0.04)
            from ml.streaming.segmenter import SegmentEvent

            event = SegmentEvent(start_time=0.8, end_time=1.6,
                                 reason=SegmentReason.SILENCE)
            frames = session.collect_segment(event)
        finally:
            session.close()
        # 0.8s of utterance plus the 0.25s lead-in, at 25 fps.
        assert 24 <= len(frames) <= 28


class TestResampling:
    def test_puts_a_30fps_capture_onto_the_25fps_grid(self) -> None:
        # A browser delivering 30 fps must not make speech play back slowly to
        # the model, which has no time base other than frame count.
        frames = _buffered([index / 30 for index in range(30)])
        resampled = _resample(frames, target_fps=25)
        assert len(resampled) == pytest.approx(25, abs=1)

    def test_passes_through_a_matching_rate(self) -> None:
        frames = _buffered([index / 25 for index in range(25)])
        assert len(_resample(frames, target_fps=25)) == 25

    def test_upsamples_a_slow_capture(self) -> None:
        frames = _buffered([index / 12.5 for index in range(13)])
        resampled = _resample(frames, target_fps=25)
        assert len(resampled) == pytest.approx(25, abs=1)

    def test_preserves_order(self) -> None:
        frames = _buffered([index / 30 for index in range(30)])
        values = [int(frame[0, 0, 0]) for frame in _resample(frames, target_fps=25)]
        assert values == sorted(values)

    def test_handles_a_single_frame(self) -> None:
        assert len(_resample(_buffered([0.0]), target_fps=25)) == 1

    def test_handles_no_frames(self) -> None:
        assert _resample([], target_fps=25) == []


class TestSegmenterIntegration:
    def test_stays_idle_when_nothing_moves(
        self, session: LiveSession, portrait: np.ndarray
    ) -> None:
        for index in range(30):
            observation = session.observe(portrait, timestamp=index * 0.04)
            assert session.update_segmenter(observation) is None
        assert session.state is SegmenterState.IDLE

    def test_flush_returns_nothing_when_idle(self, session: LiveSession) -> None:
        assert session.flush() is None

    def test_lip_motion_opens_and_closes_an_utterance(
        self, session: LiveSession, talking_frames: list[np.ndarray]
    ) -> None:
        events = []
        for index, frame in enumerate(talking_frames):
            observation = session.observe(frame, timestamp=index * 0.04)
            event = session.update_segmenter(observation)
            if event is not None:
                events.append(event)

        assert events, "a moving mouth produced no utterance"
        first = events[0]
        assert first.duration <= session.config.streaming.max_utterance_seconds + 0.1
        assert first.duration >= session.config.streaming.min_utterance_seconds
        assert first.reason is SegmentReason.MAX_LENGTH

    def test_a_moving_head_with_a_closed_mouth_is_not_an_utterance(
        self, session: LiveSession, face_frames: list[np.ndarray]
    ) -> None:
        # The reason segmentation measures lip geometry rather than pixel change:
        # moving your head alters far more of the image around the mouth than
        # speaking does, so an appearance signal transcribes every fidget.
        events = []
        measured = 0
        for index, frame in enumerate(face_frames):
            observation = session.observe(frame, timestamp=index * 0.04)
            measured += observation.energy > 0.0
            events.append(session.update_segmenter(observation))

        # Without this the test would also pass if the mouth were never found.
        assert measured > len(face_frames) // 2
        assert not any(events)
        assert session.flush() is None

    def test_a_closed_utterance_yields_frames_at_the_model_rate(
        self, session: LiveSession, talking_frames: list[np.ndarray]
    ) -> None:
        event = None
        for index, frame in enumerate(talking_frames):
            observation = session.observe(frame, timestamp=index * 0.04)
            event = session.update_segmenter(observation) or event
            if event is not None:
                break

        assert event is not None
        frames = session.collect_segment(event)
        # The lead-in is only present if the buffer reaches back that far, which
        # it does not for an utterance that starts on the first frame.
        rate = session.config.preprocessing.target_fps
        assert event.duration * rate - 2 <= len(frames) <= (event.duration + 0.25) * rate + 2
