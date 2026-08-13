"use client";

import { useEffect, type RefObject } from "react";

import type { BoxCoordinates, LiveStatus } from "@/lib/types";

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  stream: MediaStream | null;
  status: LiveStatus | null;
  /** Draw the face box and landmarks as well as the mouth. */
  detailed?: boolean;
  children?: React.ReactNode;
}

/**
 * The camera frame, with the mouth region marked.
 *
 * Boxes are positioned in percentages taken from the frame dimensions the
 * backend reports, so they stay correct at any display size and at any camera
 * resolution. The video is mirrored for the viewer, so the overlay is mirrored
 * with it — otherwise the indicator would track the wrong side of the face.
 */
export function CameraStage({ videoRef, stream, status, detailed, children }: Props) {
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.srcObject !== stream) video.srcObject = stream;
  }, [stream, videoRef]);

  const frame =
    status && status.frame_width > 0
      ? { width: status.frame_width, height: status.frame_height }
      : null;

  return (
    <div className="border-edge bg-surface relative aspect-[4/3] w-full overflow-hidden rounded-2xl border shadow-[0_30px_80px_-40px_rgba(0,0,0,0.9)]">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="mirror h-full w-full object-cover"
      />

      <div className="mirror pointer-events-none absolute inset-0">
        {detailed && status?.face_box && frame && (
          <Box
            box={status.face_box}
            frame={frame}
            className="border-faint/60 rounded-md border"
          />
        )}
        {status?.mouth_box && frame && (
          <Box
            box={status.mouth_box}
            frame={frame}
            className="border-signal/80 shadow-signal/20 rounded-[10px] border-2 shadow-[0_0_0_6px_var(--tw-shadow-color)]"
          />
        )}
        {detailed && status?.lip_contour && frame && (
          <LipContour points={status.lip_contour} frame={frame} />
        )}
      </div>

      {children}
    </div>
  );
}

/**
 * The inner lip outline the segmenter measures.
 *
 * Drawn as an SVG polygon in the frame's own coordinate system, so the browser
 * scales it with the video rather than the component recomputing percentages
 * for twenty points.
 */
function LipContour({
  points,
  frame,
}: {
  points: [number, number][];
  frame: { width: number; height: number };
}) {
  return (
    <svg
      viewBox={`0 0 ${frame.width} ${frame.height}`}
      preserveAspectRatio="none"
      className="absolute inset-0 h-full w-full"
    >
      <polygon
        points={points.map(([x, y]) => `${x},${y}`).join(" ")}
        className="fill-signal/10 stroke-signal/70"
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function Box({
  box,
  frame,
  className,
}: {
  box: BoxCoordinates;
  frame: { width: number; height: number };
  className: string;
}) {
  return (
    <div
      className={`absolute transition-all duration-150 ease-out ${className}`}
      style={{
        left: `${(box.x / frame.width) * 100}%`,
        top: `${(box.y / frame.height) * 100}%`,
        width: `${(box.width / frame.width) * 100}%`,
        height: `${(box.height / frame.height) * 100}%`,
      }}
    />
  );
}
