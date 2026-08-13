"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { STREAM_URL } from "./api";
import type {
  LiveConfig,
  LiveStatus,
  ModelSpec,
  ServerMessage,
  TranscriptionResult,
} from "./types";

export type Connection = "idle" | "connecting" | "open" | "closed" | "error";

export interface LiveTranscript extends TranscriptionResult {
  sequence: number;
  queue_wait_ms: number;
  dropped: boolean;
  receivedAt: number;
}

interface SessionInfo {
  model: string;
  modelLoaded: boolean;
  modelError: string | null;
  spec: ModelSpec;
  config: LiveConfig;
}

interface LiveSession {
  connection: Connection;
  info: SessionInfo | null;
  status: LiveStatus | null;
  latest: LiveTranscript | null;
  history: LiveTranscript[];
  /** Frames per second actually sent, measured on the client. */
  sendRate: number;
  framesSent: number;
  error: string | null;
  /** Ask the backend to transcribe whatever utterance is currently open. */
  flush: () => void;
  reset: () => void;
}

/** Frames sent for JPEG encoding; matches the backend's detection resolution. */
const CAPTURE_LONG_SIDE = 640;
const JPEG_QUALITY = 0.72;

/**
 * Skip a frame rather than queue it once this much is already waiting on the
 * socket. Sending anyway would build a backlog that shows up as the transcript
 * lagging further behind the speaker with every second.
 */
const MAX_BUFFERED_BYTES = 512 * 1024;

const HISTORY_LIMIT = 12;

export function useLiveSession(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  enabled: boolean,
): LiveSession {
  // Held as the socket's own phase and combined with `enabled` at read time, so
  // that turning the camera off does not need a state update to take effect.
  const [phase, setPhase] = useState<Connection>("connecting");
  const connection: Connection = enabled ? phase : "idle";

  const [info, setInfo] = useState<SessionInfo | null>(null);
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [latest, setLatest] = useState<LiveTranscript | null>(null);
  const [history, setHistory] = useState<LiveTranscript[]>([]);
  const [sendRate, setSendRate] = useState(0);
  const [framesSent, setFramesSent] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const encodingRef = useRef(false);
  const sentInWindowRef = useRef(0);

  const send = useCallback((payload: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }, []);

  const flush = useCallback(() => send({ type: "flush" }), [send]);
  const reset = useCallback(() => {
    setLatest(null);
    send({ type: "reset" });
  }, [send]);

  useEffect(() => {
    if (!enabled) return;

    let disposed = false;
    const socket = new WebSocket(STREAM_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => !disposed && setPhase("open");

    socket.onmessage = (event) => {
      if (disposed || typeof event.data !== "string") return;
      const message = JSON.parse(event.data) as ServerMessage;

      switch (message.type) {
        case "ready":
          setInfo({
            model: message.model,
            modelLoaded: message.model_loaded,
            modelError: message.error,
            spec: message.spec,
            config: message.config,
          });
          break;
        case "status":
          setStatus(message);
          break;
        case "transcript": {
          const entry: LiveTranscript = { ...message, receivedAt: Date.now() };
          if (!entry.dropped) setLatest(entry);
          setHistory((previous) => [entry, ...previous].slice(0, HISTORY_LIMIT));
          break;
        }
        case "error":
          setError(message.message);
          break;
      }
    };

    socket.onerror = () => {
      if (disposed) return;
      setPhase("error");
      setError("Lost the connection to the speech service.");
    };

    socket.onclose = () => !disposed && setPhase("closed");

    return () => {
      disposed = true;
      socketRef.current = null;
      if (socket.readyState === WebSocket.OPEN) socket.close(1000, "client stopped");
      else socket.close();

      // Everything below describes a session that no longer exists. Clearing it
      // here, rather than leaving it on screen, stops a stale mouth box from
      // hovering over a stopped camera.
      setPhase("connecting");
      setStatus(null);
      setLatest(null);
      setHistory([]);
      setFramesSent(0);
      setError(null);
    };
  }, [enabled]);

  // Frame pump. Capture is paced to the model's frame rate rather than the
  // display's: the model has no other time base, and sending 60 fps would make
  // speech read as twice as fast as it was spoken.
  const targetFps = info?.config.target_fps ?? 25;
  useEffect(() => {
    if (!enabled || connection !== "open") return;

    const interval = window.setInterval(() => {
      const video = videoRef.current;
      const socket = socketRef.current;
      if (!video || !socket || socket.readyState !== WebSocket.OPEN) return;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      if (encodingRef.current || socket.bufferedAmount > MAX_BUFFERED_BYTES) return;

      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      if (!sourceWidth || !sourceHeight) return;

      const scale = Math.min(1, CAPTURE_LONG_SIDE / Math.max(sourceWidth, sourceHeight));
      const width = Math.round(sourceWidth * scale);
      const height = Math.round(sourceHeight * scale);

      let canvas = canvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
        canvasRef.current = canvas;
      }
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const context = canvas.getContext("2d", { alpha: false });
      if (!context) return;
      context.drawImage(video, 0, 0, width, height);

      encodingRef.current = true;
      canvas.toBlob(
        (blob) => {
          encodingRef.current = false;
          if (!blob || socket.readyState !== WebSocket.OPEN) return;
          blob.arrayBuffer().then((buffer) => {
            if (socket.readyState !== WebSocket.OPEN) return;
            socket.send(buffer);
            sentInWindowRef.current += 1;
            setFramesSent((count) => count + 1);
          });
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }, 1000 / targetFps);

    const rateTimer = window.setInterval(() => {
      setSendRate(sentInWindowRef.current);
      sentInWindowRef.current = 0;
    }, 1000);

    return () => {
      window.clearInterval(interval);
      window.clearInterval(rateTimer);
      setSendRate(0);
    };
  }, [enabled, connection, targetFps, videoRef]);

  return {
    connection,
    info,
    status,
    latest,
    history,
    sendRate,
    framesSent,
    error,
    flush,
    reset,
  };
}
