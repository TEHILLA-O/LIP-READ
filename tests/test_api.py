"""API endpoint tests.

Run against the ``custom`` adapter, which is a deliberate stub. That gives a
service that starts in a second without a 1 GB checkpoint and still exercises
everything up to the model boundary: uploads, preprocessing, the result
contract, debug artefacts and temp-file cleanup. Transcription quality is
covered separately by the LRS3 evaluation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def temp_store_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("api-transient")


@pytest.fixture(scope="module")
def client(temp_store_dir: Path) -> Iterator[TestClient]:
    previous = {
        key: os.environ.get(key) for key in ("VSR_MODEL", "VSR_TEMP_DIR", "VSR_DEVICE")
    }
    os.environ["VSR_MODEL"] = "custom"
    os.environ["VSR_TEMP_DIR"] = str(temp_store_dir)
    os.environ["VSR_DEVICE"] = "cpu"
    try:
        from app.main import create_app

        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def live_client(temp_store_dir: Path) -> Iterator[TestClient]:
    """A service with a short minimum utterance.

    The live path timestamps frames as they arrive, so with the default 1.2 s
    minimum whether a segment closes would depend on how fast the test machine
    pushes frames. Shortening it makes the assertion about behaviour rather than
    about timing.
    """
    keys = ("VSR_MODEL", "VSR_TEMP_DIR", "VSR_DEVICE", "VSR_MIN_UTTERANCE_SECONDS")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(
        VSR_MODEL="custom",
        VSR_TEMP_DIR=str(temp_store_dir),
        VSR_DEVICE="cpu",
        VSR_MIN_UTTERANCE_SECONDS="0.2",
    )
    try:
        from app.main import create_app

        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _await_message(socket, kind: str, limit: int = 40) -> dict:
    """Read until a message of ``kind`` arrives, ignoring status chatter."""
    for _ in range(limit):
        message = socket.receive_json()
        if message["type"] == kind:
            return message
    raise AssertionError(f"No {kind!r} message within {limit} messages")


def _upload(path: Path) -> dict:
    return {"file": (path.name, path.read_bytes(), "video/mp4")}


class TestHealth:
    def test_reports_ok_even_without_a_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["device"] == "cpu"

    def test_explains_why_the_model_is_unavailable(self, client: TestClient) -> None:
        # A stub adapter must say so, not silently behave like a loaded model.
        assert client.get("/health").json()["error"]

    def test_lists_registered_adapters(self, client: TestClient) -> None:
        body = client.get("/v1/models").json()
        assert body["active"] == "custom"
        assert body["available"]["auto_avsr"] is True
        assert body["available"]["vallr"] is False

    def test_publishes_the_input_contract(self, client: TestClient) -> None:
        spec = client.get("/v1/models").json()["spec"]
        assert spec["fps"] == 25
        assert spec["input_size"] == [88, 88]
        assert spec["channels"] == 1


class TestPreprocessEndpoint:
    def test_extracts_a_mouth_track(self, client: TestClient, still_clip: Path) -> None:
        body = client.post("/v1/preprocess", files=_upload(still_clip)).json()
        assert body["mouth_detected"] is True
        assert body["detection_ratio"] > 0.9
        assert body["frames_processed"] == 30
        assert body["roi_shape"] == [30, 96, 96, 3]

    def test_converts_the_frame_rate(self, client: TestClient, clip_30fps: Path) -> None:
        body = client.post("/v1/preprocess", files=_upload(clip_30fps)).json()
        assert body["source_fps"] == pytest.approx(30.0, abs=0.1)
        assert body["target_fps"] == 25
        # 60 source frames at 30 fps is two seconds, which is 50 at 25 fps.
        assert body["frames_processed"] == pytest.approx(50, abs=1)

    def test_reports_a_faceless_clip_without_failing(
        self, client: TestClient, no_face_clip: Path
    ) -> None:
        response = client.post("/v1/preprocess", files=_upload(no_face_clip))
        assert response.status_code == 200
        body = response.json()
        assert body["mouth_detected"] is False
        assert "no face" in body["error"].lower()

    def test_debug_clip_is_downloadable(
        self, client: TestClient, still_clip: Path
    ) -> None:
        body = client.post(
            "/v1/preprocess?debug=true", files=_upload(still_clip)
        ).json()
        url = body["debug"]["mouth_roi"]
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 0

    def test_expired_debug_clip_says_so(self, client: TestClient) -> None:
        response = client.get("/v1/debug/" + "0" * 32 + "/mouth_roi")
        assert response.status_code == 404
        assert "expired" in response.json()["detail"]


class TestTranscribeEndpoint:
    def test_returns_the_full_result_contract(
        self, client: TestClient, still_clip: Path
    ) -> None:
        body = client.post("/v1/transcribe", files=_upload(still_clip)).json()
        for key in (
            "text",
            "confidence",
            "alternatives",
            "processing_ms",
            "frames_processed",
            "mouth_detected",
        ):
            assert key in body

    def test_states_that_the_model_is_unavailable(
        self, client: TestClient, still_clip: Path
    ) -> None:
        # The critical guarantee: no transcript is invented when no model exists.
        body = client.post("/v1/transcribe", files=_upload(still_clip)).json()
        assert body["text"] == ""
        assert body["confidence"] == 0.0
        assert body["error"]
        assert body["mouth_detected"] is True
        assert body["frames_processed"] == 30

    def test_rejects_a_non_video_upload(self, client: TestClient) -> None:
        response = client.post(
            "/v1/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 415

    def test_rejects_an_empty_upload(self, client: TestClient) -> None:
        response = client.post(
            "/v1/transcribe", files={"file": ("empty.mp4", b"", "video/mp4")}
        )
        assert response.status_code == 400

    def test_rejects_an_unreadable_video(self, client: TestClient) -> None:
        response = client.post(
            "/v1/transcribe", files={"file": ("broken.mp4", b"\x00" * 4096, "video/mp4")}
        )
        assert response.status_code == 422

    def test_enforces_the_upload_size_limit(
        self, client: TestClient, still_clip: Path
    ) -> None:
        from app.settings import Settings, get_settings

        small = Settings(max_upload_bytes=1024)
        client.app.dependency_overrides[get_settings] = lambda: small
        try:
            response = client.post("/v1/transcribe", files=_upload(still_clip))
        finally:
            client.app.dependency_overrides.clear()
        assert response.status_code == 413


class TestTemporaryFileCleanup:
    def test_upload_is_deleted_after_inference(
        self, client: TestClient, temp_store_dir: Path, still_clip: Path
    ) -> None:
        client.post("/v1/transcribe", files=_upload(still_clip))
        assert list(temp_store_dir.glob("upload-*")) == []

    def test_upload_is_deleted_after_a_failure(
        self, client: TestClient, temp_store_dir: Path
    ) -> None:
        # The file is written before decoding fails, so only a `finally` saves it.
        client.post(
            "/v1/transcribe", files={"file": ("broken.mp4", b"\x00" * 4096, "video/mp4")}
        )
        assert list(temp_store_dir.glob("upload-*")) == []

    def test_debug_artefacts_are_swept_when_stale(
        self, client: TestClient, temp_store_dir: Path, still_clip: Path
    ) -> None:
        from ml.utils.tempfiles import TransientStore

        client.post("/v1/preprocess?debug=true", files=_upload(still_clip))
        assert list(temp_store_dir.glob("debug-*"))
        TransientStore(temp_store_dir).sweep(max_age_seconds=-1)
        assert list(temp_store_dir.glob("debug-*")) == []


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


class TestWebSocket:
    def test_announces_the_model_and_config_on_connect(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert ready["model_loaded"] is False
        assert ready["config"]["target_fps"] == 25

    def test_reports_mouth_detection_for_streamed_frames(
        self, client: TestClient, face_frames: list[np.ndarray]
    ) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            for frame in face_frames[:5]:
                socket.send_bytes(_jpeg(frame))
            status = socket.receive_json()
        assert status["type"] == "status"
        assert status["mouth_detected"] is True
        assert status["face_box"]["width"] > 0
        assert status["buffer_frames"] == 5

    def test_survives_an_undecodable_frame(
        self, client: TestClient, face_frames: list[np.ndarray]
    ) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            socket.send_bytes(b"this is not an image")
            for frame in face_frames[:5]:
                socket.send_bytes(_jpeg(frame))
            status = socket.receive_json()
        assert status["frames_rejected"] == 1
        assert status["frames_received"] == 5

    def test_responds_to_ping(self, client: TestClient) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            socket.send_json({"type": "ping"})
            assert socket.receive_json()["type"] == "pong"

    def test_reset_clears_the_buffer(
        self, client: TestClient, face_frames: list[np.ndarray]
    ) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            for frame in face_frames[:5]:
                socket.send_bytes(_jpeg(frame))
            socket.receive_json()
            socket.send_json({"type": "reset"})
            assert socket.receive_json()["type"] == "reset"

            for frame in face_frames[5:10]:
                socket.send_bytes(_jpeg(frame))
            status = socket.receive_json()
        assert status["buffer_frames"] == 5

    def test_flush_with_nothing_open_is_not_an_error(self, client: TestClient) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            socket.send_json({"type": "flush"})
            message = socket.receive_json()
        assert message["type"] == "segment"
        assert message["accepted"] is False

    def test_rejects_an_unknown_command(self, client: TestClient) -> None:
        with client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            socket.send_json({"type": "teleport"})
            message = socket.receive_json()
        assert message["type"] == "error"


class TestLiveTranscription:
    """The whole live path over a real socket: frames in, transcript out."""

    def test_a_moving_mouth_becomes_a_transcribed_segment(
        self, live_client: TestClient, talking_frames: list[np.ndarray]
    ) -> None:
        with live_client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            for frame in talking_frames[:60]:
                socket.send_bytes(_jpeg(frame))
            socket.send_json({"type": "flush"})

            segment = _await_message(socket, "segment")
            assert segment["accepted"] is True
            assert segment["frames"] > 0
            assert segment["displaced_earlier_segment"] is False

            transcript = _await_message(socket, "transcript")

        assert transcript["sequence"] == segment["sequence"]
        # The stub adapter cannot transcribe, and must say so rather than
        # returning an empty string that would read as silence.
        assert transcript["text"] == ""
        assert transcript["confidence"] == 0.0
        assert transcript["error"]
        assert set(transcript) >= {
            "text", "confidence", "alternatives",
            "processing_ms", "frames_processed", "mouth_detected",
        }

    def test_a_still_face_never_opens_a_segment(
        self, live_client: TestClient, portrait: np.ndarray
    ) -> None:
        encoded = _jpeg(portrait)
        with live_client.websocket_connect("/v1/stream") as socket:
            socket.receive_json()
            for _ in range(40):
                socket.send_bytes(encoded)
            socket.send_json({"type": "flush"})
            message = _await_message(socket, "segment")
        assert message["accepted"] is False
        assert message["reason"] == "nothing_pending"
