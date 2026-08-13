"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type CameraState = "idle" | "starting" | "live" | "denied" | "error";

interface Camera {
  state: CameraState;
  error: string | null;
  stream: MediaStream | null;
  /** Native resolution of the track, needed to map overlay coordinates. */
  dimensions: { width: number; height: number } | null;
  start: () => Promise<void>;
  stop: () => void;
}

const CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
    facingMode: "user",
  },
  audio: false,
};

function describe(error: unknown): { state: CameraState; message: string } {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return {
        state: "denied",
        message:
          "Camera access was blocked. Allow it in your browser's site settings, then try again.",
      };
    }
    if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
      return { state: "error", message: "No camera was found on this device." };
    }
    if (error.name === "NotReadableError") {
      return {
        state: "error",
        message: "The camera is already in use by another application.",
      };
    }
  }
  return { state: "error", message: "The camera could not be started." };
}

/** Owns the getUserMedia lifecycle, including teardown on unmount. */
export function useCamera(): Camera {
  const [state, setState] = useState<CameraState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(
    null,
  );
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
    setDimensions(null);
    setState("idle");
  }, []);

  const start = useCallback(async () => {
    if (streamRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setState("error");
      setError("This browser does not support camera capture.");
      return;
    }

    setState("starting");
    setError(null);
    try {
      const media = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
      streamRef.current = media;
      setStream(media);

      const settings = media.getVideoTracks()[0]?.getSettings();
      setDimensions({
        width: settings?.width ?? 640,
        height: settings?.height ?? 480,
      });
      setState("live");
    } catch (caught) {
      const described = describe(caught);
      setState(described.state);
      setError(described.message);
    }
  }, []);

  // Releasing the camera on unmount matters for more than tidiness: the
  // recording light staying on after the user navigates away is alarming.
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  return { state, error, stream, dimensions, start, stop };
}
