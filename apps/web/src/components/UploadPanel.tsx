"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "./Controls";

interface Props {
  file: File | null;
  onSelect: (file: File | null) => void;
  onTranscribe: () => void;
  busy: boolean;
  disabled?: boolean;
}

const ACCEPTED = "video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.webm";

/** File selection and preview. The transcript itself is rendered by the page. */
export function UploadPanel({ file, onSelect, onTranscribe, busy, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    // Object URLs pin the whole file in memory until they are revoked.
    if (preview) return () => URL.revokeObjectURL(preview);
  }, [preview]);

  return (
    <div className="space-y-6">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const dropped = event.dataTransfer.files?.[0];
          if (dropped) onSelect(dropped);
        }}
        className={`relative aspect-[4/3] w-full overflow-hidden rounded-2xl border transition-colors duration-200 ${
          dragging ? "border-signal bg-signal/5" : "border-edge bg-surface"
        }`}
      >
        {preview ? (
          <video
            key={preview}
            src={preview}
            controls
            playsInline
            className="h-full w-full object-contain"
          />
        ) : (
          <button
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
            className="text-muted hover:text-chalk flex h-full w-full cursor-pointer flex-col items-center justify-center gap-3 transition-colors disabled:cursor-not-allowed"
          >
            <span className="border-edge flex h-12 w-12 items-center justify-center rounded-full border">
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
                <path
                  d="M9 12V3m0 0L5.5 6.5M9 3l3.5 3.5M3 12v2a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2"
                  stroke="currentColor"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <span className="text-xs tracking-[0.2em] uppercase">Choose a video</span>
            <span className="text-faint text-xs normal-case">
              or drop one here &middot; mp4, mov, webm
            </span>
          </button>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
      />

      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button onClick={() => inputRef.current?.click()} disabled={busy || disabled}>
          {file ? "Change" : "Choose"}
        </Button>
        <Button variant="primary" onClick={onTranscribe} disabled={!file || busy || disabled}>
          {busy ? "Reading lips" : "Transcribe"}
        </Button>
        {file && !busy && (
          <Button variant="ghost" onClick={() => onSelect(null)}>
            Clear
          </Button>
        )}
      </div>

      {file && (
        <p className="text-faint text-center text-xs">
          {file.name} &middot; {(file.size / (1024 * 1024)).toFixed(1)} MB
        </p>
      )}
    </div>
  );
}
