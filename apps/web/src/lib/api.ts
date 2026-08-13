import type { Health, TranscriptionResult } from "./types";

/**
 * The ML backend runs as a separate service, so its origin is configuration
 * rather than a relative path.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_ML_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export const STREAM_URL = `${API_BASE.replace(/^http/, "ws")}/v1/stream`;

export function debugUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : response.statusText;
  } catch {
    return response.statusText || `Request failed with ${response.status}`;
  }
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch(`${API_BASE}/health`, { signal, cache: "no-store" });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export async function transcribeFile(
  file: File,
  options: { debug?: boolean; signal?: AbortSignal } = {},
): Promise<TranscriptionResult> {
  const body = new FormData();
  body.append("file", file);

  const query = options.debug ? "?debug=true" : "";
  const response = await fetch(`${API_BASE}/v1/transcribe${query}`, {
    method: "POST",
    body,
    signal: options.signal,
  });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}
