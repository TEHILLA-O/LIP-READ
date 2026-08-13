"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraStage } from "@/components/CameraStage";
import { Button, Indicator, ModeSwitch, type Mode } from "@/components/Controls";
import { DebugPanel } from "@/components/DebugPanel";
import { Transcript } from "@/components/Transcript";
import { UploadPanel } from "@/components/UploadPanel";
import { fetchHealth, transcribeFile } from "@/lib/api";
import type { Health, TranscriptionResult } from "@/lib/types";
import { useCamera } from "@/lib/useCamera";
import { useLiveSession } from "@/lib/useLiveSession";

export default function Page() {
  const [mode, setMode] = useState<Mode>("camera");
  const [debug, setDebug] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const camera = useCamera();
  const live = useLiveSession(videoRef, mode === "camera" && camera.state === "live");

  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<TranscriptionResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((next) => !controller.signal.aborted && setHealth(next))
      // An abort is this component going away, not the service being down.
      // Reporting it as an outage would leave a remounted page claiming the
      // backend is offline while it is answering normally.
      .catch(() => !controller.signal.aborted && setHealth(null));
    return () => controller.abort();
  }, []);

  // Shift+D rather than a visible control: the debug view is for whoever is
  // working on the pipeline, and a button for it would clutter the main screen.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.shiftKey && event.key.toLowerCase() === "d") setDebug((on) => !on);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const switchMode = useCallback(
    (next: Mode) => {
      if (next === "upload") camera.stop();
      setMode(next);
    },
    [camera],
  );

  const runUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      setUploadResult(await transcribeFile(file, { debug }));
    } catch (caught) {
      setUploadError(
        caught instanceof Error ? caught.message : "The video could not be transcribed.",
      );
    } finally {
      setUploading(false);
    }
  }, [file, debug]);

  const serviceDown = health === null;
  const modelMissing = health !== null && !health.model_loaded;
  const result = mode === "camera" ? live.latest : uploadResult;

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col px-6 py-14 sm:py-20">
      <header className="mb-12 text-center">
        <h1 className="text-muted text-xs tracking-[0.42em] uppercase">
          Visual Speech
        </h1>
      </header>

      <section className="space-y-8">
        {mode === "camera" ? (
          <CameraStage
            videoRef={videoRef}
            stream={camera.stream}
            status={live.status}
            detailed={debug}
          >
            {camera.state !== "live" && (
              <div className="bg-ink/70 absolute inset-0 flex flex-col items-center justify-center gap-5 backdrop-blur-sm">
                <p className="text-muted max-w-xs text-center text-sm leading-relaxed text-balance">
                  {camera.error ??
                    "Your camera stays on this machine. Frames are read for lip movement and never stored."}
                </p>
                <Button
                  variant="primary"
                  onClick={camera.start}
                  disabled={camera.state === "starting" || serviceDown}
                >
                  {camera.state === "starting" ? "Starting" : "Start camera"}
                </Button>
              </div>
            )}
          </CameraStage>
        ) : (
          <UploadPanel
            file={file}
            onSelect={(next) => {
              setFile(next);
              setUploadResult(null);
              setUploadError(null);
            }}
            onTranscribe={runUpload}
            busy={uploading}
            disabled={serviceDown}
          />
        )}

        <StatusLine
          mode={mode}
          serviceDown={serviceDown}
          modelMissing={modelMissing}
          cameraLive={camera.state === "live"}
          mouthDetected={live.status?.mouth_detected ?? false}
          speaking={live.status?.state === "speaking"}
          uploading={uploading}
        />

        <div className="min-h-[9rem] pt-2">
          <Transcript
            text={result?.text ?? ""}
            confidence={result?.confidence ?? 0}
            alternatives={result?.alternatives}
            pending={uploading}
            notice={
              serviceDown
                ? "The speech service is not reachable. Start it with: uvicorn app.main:app --port 8000"
                : (uploadError ??
                  (health && !health.model_loaded
                    ? `Transcription is unavailable: ${health.error ?? "no model is loaded"}`
                    : (result?.error ?? null)))
            }
            placeholder={
              mode === "camera"
                ? camera.state === "live"
                  ? "Speak clearly, facing the camera."
                  : ""
                : "Choose a video to read."
            }
          />
        </div>
      </section>

      <div className="mt-auto flex flex-col items-center gap-5 pt-14">
        <ModeSwitch mode={mode} onChange={switchMode} />

        {mode === "camera" && camera.state === "live" && (
          <div className="flex gap-3">
            <Button onClick={live.flush} disabled={live.connection !== "open"}>
              Transcribe now
            </Button>
            <Button variant="ghost" onClick={camera.stop}>
              Stop
            </Button>
          </div>
        )}

        <button
          onClick={() => setDebug((on) => !on)}
          className="text-faint hover:text-muted cursor-pointer text-[10px] tracking-[0.22em] uppercase transition-colors"
        >
          {debug ? "Hide" : "Show"} debug
        </button>
      </div>

      {debug && (
        <div className="mt-8">
          <DebugPanel
            health={health}
            status={live.status}
            sendRate={live.sendRate}
            framesSent={live.framesSent}
            result={result}
            history={live.history}
          />
        </div>
      )}
    </main>
  );
}

function StatusLine({
  mode,
  serviceDown,
  modelMissing,
  cameraLive,
  mouthDetected,
  speaking,
  uploading,
}: {
  mode: Mode;
  serviceDown: boolean;
  modelMissing: boolean;
  cameraLive: boolean;
  mouthDetected: boolean;
  speaking: boolean;
  uploading: boolean;
}) {
  if (serviceDown) return <Indicator tone="alert" label="Service offline" />;
  if (modelMissing) return <Indicator tone="warn" label="No model loaded" />;

  if (mode === "upload") {
    return uploading ? (
      <Indicator tone="signal" label="Reading lips" pulse />
    ) : (
      <Indicator tone="idle" label="Ready" />
    );
  }

  if (!cameraLive) return <Indicator tone="idle" label="Camera off" />;
  if (speaking) return <Indicator tone="signal" label="Listening" pulse />;
  if (mouthDetected) return <Indicator tone="signal" label="Mouth detected" />;
  return <Indicator tone="warn" label="No mouth detected" />;
}
