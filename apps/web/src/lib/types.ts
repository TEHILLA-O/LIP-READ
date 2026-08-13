/**
 * Types mirroring the ML API's response contract.
 *
 * A result always carries `confidence` and `mouth_detected` so the interface
 * cannot present a model prediction as a certainty.
 */

export interface Alternative {
  text: string;
  confidence: number;
}

export interface TranscriptionResult {
  text: string;
  confidence: number;
  alternatives: Alternative[];
  processing_ms: number;
  frames_processed: number;
  mouth_detected: boolean;
  error?: string | null;
  diagnostics?: Diagnostics | null;
  debug?: Record<string, string> | null;
}

export interface Diagnostics {
  inference_ms?: number;
  detection_ratio?: number;
  duration_seconds?: number;
  source_fps?: number;
  target_fps?: number;
  beam_score?: number;
  token_count?: number;
  raw_output?: Record<string, unknown>;
}

export interface ModelSpec {
  name: string;
  fps: number;
  crop_size: [number, number];
  input_size: [number, number];
  channels: number;
  tensor_layout: string;
  checkpoint_required: boolean;
  description: string;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  device: string;
  model_name: string;
  error?: string | null;
}

export interface BoxCoordinates {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LiveConfig {
  target_fps: number;
  min_utterance_seconds: number;
  max_utterance_seconds: number;
  buffer_seconds: number;
}

export interface QueueStats {
  submitted: number;
  completed: number;
  dropped: number;
  queue_depth: number;
}

export type ServerMessage =
  | {
      type: "ready";
      model: string;
      model_loaded: boolean;
      error: string | null;
      spec: ModelSpec;
      config: LiveConfig;
    }
  | ({
      type: "status";
      mouth_detected: boolean;
      energy: number;
      face_box: BoxCoordinates | null;
      mouth_box: BoxCoordinates | null;
      /** Inner lip outline in source pixels, for the debug overlay. */
      lip_contour: [number, number][] | null;
      frame_width: number;
      frame_height: number;
      frames_received: number;
      frames_rejected: number;
      buffer_frames: number;
      buffer_seconds: number;
      observed_fps: number;
      state: "idle" | "speaking";
      utterance_seconds: number;
      queue: QueueStats;
    })
  | {
      type: "segment";
      sequence: number;
      reason: string;
      duration: number;
      frames: number;
      accepted: boolean;
      displaced_earlier_segment: boolean;
    }
  | ({
      type: "transcript";
      sequence: number;
      queue_wait_ms: number;
      dropped: boolean;
    } & TranscriptionResult)
  | { type: "error"; message: string }
  | { type: "pong" }
  | { type: "reset" };

export type LiveStatus = Extract<ServerMessage, { type: "status" }>;
