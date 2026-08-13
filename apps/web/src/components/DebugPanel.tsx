"use client";

import { debugUrl } from "@/lib/api";
import type { LiveTranscript } from "@/lib/useLiveSession";
import type { Health, LiveStatus, TranscriptionResult } from "@/lib/types";

type Tone = "signal" | "warn" | "alert" | "faint";

// Spelled out rather than interpolated, so Tailwind can see the class names.
const TONE_CLASS: Record<Tone, string> = {
  signal: "text-signal",
  warn: "text-warn",
  alert: "text-alert",
  faint: "text-faint",
};

interface Props {
  health: Health | null;
  status: LiveStatus | null;
  sendRate: number;
  framesSent: number;
  result: TranscriptionResult | null;
  history: LiveTranscript[];
}

/**
 * Developer view: everything the pipeline knows, laid out flat.
 *
 * Exists because visual speech fails quietly. A wrong crop, a stalled queue and
 * a genuinely hard utterance all look identical from the transcript alone, and
 * only the numbers here tell them apart.
 */
export function DebugPanel({
  health,
  status,
  sendRate,
  framesSent,
  result,
  history,
}: Props) {
  const diagnostics = result?.diagnostics ?? null;
  const raw = diagnostics?.raw_output as Record<string, unknown> | undefined;

  return (
    <div className="border-edge bg-surface/60 space-y-6 rounded-2xl border p-6 text-xs">
      <Section title="Service">
        <Row label="Model" value={health?.model_name ?? "-"} />
        <Row label="Device" value={health?.device ?? "-"} />
        <Row
          label="Weights"
          value={health?.model_loaded ? "loaded" : (health?.error ?? "not loaded")}
          tone={health?.model_loaded ? "signal" : "alert"}
        />
      </Section>

      <Section title="Capture">
        <Row label="Frames sent" value={framesSent} />
        <Row label="Send rate" value={`${sendRate} fps`} />
        <Row
          label="Server rate"
          value={status ? `${status.observed_fps.toFixed(1)} fps` : "-"}
        />
        <Row
          label="Frame size"
          value={status ? `${status.frame_width}x${status.frame_height}` : "-"}
        />
        <Row label="Rejected" value={status?.frames_rejected ?? 0} />
      </Section>

      <Section title="Detection">
        <Row
          label="Mouth"
          value={status?.mouth_detected ? "detected" : "not detected"}
          tone={status?.mouth_detected ? "signal" : "faint"}
        />
        <Row label="Face box" value={format(status?.face_box)} />
        <Row label="Mouth box" value={format(status?.mouth_box)} />
        {/* The mesh can fail on a frame the face detector handles, which stops
            segmentation without stopping the mouth indicator. */}
        <Row
          label="Lip mesh"
          value={status?.lip_contour ? `${status.lip_contour.length} points` : "no fit"}
          tone={status?.lip_contour ? "signal" : "warn"}
        />
        <Row label="Motion energy" value={status?.energy.toFixed(4) ?? "-"} />
      </Section>

      <Section title="Buffer and queue">
        <Row
          label="Buffer"
          value={
            status ? `${status.buffer_frames} frames / ${status.buffer_seconds}s` : "-"
          }
        />
        <Row label="Segmenter" value={status?.state ?? "-"} />
        <Row label="Utterance" value={`${status?.utterance_seconds ?? 0}s`} />
        <Row label="Queue depth" value={status?.queue.queue_depth ?? 0} />
        <Row
          label="Dropped"
          value={status?.queue.dropped ?? 0}
          tone={status && status.queue.dropped > 0 ? "warn" : undefined}
        />
        <Row label="Completed" value={status?.queue.completed ?? 0} />
      </Section>

      <Section title="Last inference">
        <Row label="Decoded text" value={result?.text || "-"} />
        <Row
          label="Confidence"
          value={result ? result.confidence.toFixed(4) : "-"}
        />
        <Row label="Frames" value={result?.frames_processed ?? "-"} />
        <Row label="Total" value={result ? `${result.processing_ms} ms` : "-"} />
        <Row
          label="Inference"
          value={diagnostics?.inference_ms ? `${diagnostics.inference_ms} ms` : "-"}
        />
        <Row
          label="Detection ratio"
          value={
            diagnostics?.detection_ratio !== undefined
              ? `${Math.round(diagnostics.detection_ratio * 100)}%`
              : "-"
          }
        />
        <Row label="Beam score" value={diagnostics?.beam_score ?? "-"} />
        <Row label="Tokens" value={diagnostics?.token_count ?? "-"} />
        {result?.error && <Row label="Error" value={result.error} tone="alert" />}
      </Section>

      {raw && (
        <Section title="Raw model output">
          {Object.entries(raw).map(([key, value]) => (
            <Row key={key} label={key} value={JSON.stringify(value)} />
          ))}
        </Section>
      )}

      {result?.alternatives && result.alternatives.length > 0 && (
        <Section title="Alternatives">
          {result.alternatives.map((alternative) => (
            <Row
              key={alternative.text}
              label={alternative.confidence.toFixed(3)}
              value={alternative.text}
            />
          ))}
        </Section>
      )}

      {result?.debug && (
        <Section title="Processed mouth video">
          <p className="text-faint mb-2 normal-case">
            What the model was actually given. Deleted from the server on a timer.
          </p>
          {Object.entries(result.debug).map(([kind, path]) => (
            <a
              key={kind}
              href={debugUrl(path)}
              target="_blank"
              rel="noreferrer"
              className="text-signal hover:text-chalk block font-mono underline underline-offset-4"
            >
              {kind}.mp4
            </a>
          ))}
        </Section>
      )}

      {history.length > 0 && (
        <Section title="Recent segments">
          {history.map((entry) => (
            <Row
              key={`${entry.sequence}-${entry.receivedAt}`}
              label={`#${entry.sequence} ${entry.queue_wait_ms}ms wait`}
              value={entry.dropped ? "dropped" : entry.text || entry.error || "-"}
              tone={entry.dropped ? "warn" : undefined}
            />
          ))}
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-faint mb-2 text-[10px] tracking-[0.22em] uppercase">
        {title}
      </h3>
      <dl className="space-y-1">{children}</dl>
    </div>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  tone?: Tone;
}) {
  const colour = tone ? TONE_CLASS[tone] : "text-chalk";
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-muted shrink-0">{label}</dt>
      <dd className={`truncate text-right font-mono ${colour}`}>{value}</dd>
    </div>
  );
}

function format(box: { x: number; y: number; width: number; height: number } | null | undefined) {
  return box ? `${box.x},${box.y} ${box.width}x${box.height}` : "-";
}
