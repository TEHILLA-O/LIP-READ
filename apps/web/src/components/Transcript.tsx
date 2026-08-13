"use client";

import type { Alternative } from "@/lib/types";

interface Props {
  text: string;
  confidence: number;
  alternatives?: Alternative[];
  /** Shown instead of a transcript when there is nothing to say yet. */
  placeholder?: string;
  notice?: string | null;
  pending?: boolean;
}

/**
 * The transcript, which dominates the page below the camera.
 *
 * A prediction is never presented as fact: the confidence sits with the text,
 * and low-confidence output is visibly dimmed rather than silently shown as if
 * it were certain.
 */
export function Transcript({
  text,
  confidence,
  alternatives = [],
  placeholder,
  notice,
  pending,
}: Props) {
  if (notice) {
    return (
      <p className="text-muted mx-auto max-w-xl text-center text-base leading-relaxed text-balance">
        {notice}
      </p>
    );
  }

  if (!text) {
    return (
      <p className="text-faint text-center text-lg">
        {pending ? <Ellipsis /> : placeholder}
      </p>
    );
  }

  const weak = confidence < 0.4;

  return (
    <div key={text} className="animate-rise space-y-5 text-center">
      <p
        className={`mx-auto max-w-3xl text-3xl leading-tight font-light tracking-tight text-balance sm:text-4xl ${
          weak ? "text-chalk/55" : "text-chalk"
        }`}
      >
        &ldquo;{sentenceCase(text)}&rdquo;
      </p>

      <div className="text-muted flex items-center justify-center gap-3 text-xs tracking-[0.18em] uppercase">
        <span>confidence {Math.round(confidence * 100)}%</span>
        <Meter value={confidence} />
      </div>

      {alternatives.length > 0 && (
        <ul className="text-faint space-y-1 text-sm">
          {alternatives.slice(0, 2).map((alternative) => (
            <li key={alternative.text}>
              {sentenceCase(alternative.text)}
              <span className="text-faint/70 ml-2 text-xs">
                {Math.round(alternative.confidence * 100)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Meter({ value }: { value: number }) {
  const tone = value >= 0.6 ? "bg-signal" : value >= 0.4 ? "bg-warn" : "bg-alert";
  return (
    <span className="bg-edge h-px w-24 overflow-hidden rounded-full">
      <span
        className={`block h-full transition-[width] duration-500 ${tone}`}
        style={{ width: `${Math.max(4, Math.round(value * 100))}%` }}
      />
    </span>
  );
}

function Ellipsis() {
  return (
    <span className="inline-flex gap-1.5" aria-label="Transcribing">
      {[0, 1, 2].map((index) => (
        <span
          key={index}
          className="bg-faint animate-breathe inline-block h-1.5 w-1.5 rounded-full"
          style={{ animationDelay: `${index * 0.18}s` }}
        />
      ))}
    </span>
  );
}

/** The model emits upper case; sentence case reads as speech rather than a label. */
function sentenceCase(text: string): string {
  const lower = text.toLocaleLowerCase();
  return lower.charAt(0).toLocaleUpperCase() + lower.slice(1);
}
