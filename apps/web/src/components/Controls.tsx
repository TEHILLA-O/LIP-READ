"use client";

export type Mode = "camera" | "upload";

/** Small, quiet controls. Nothing here should compete with the transcript. */
export function ModeSwitch({
  mode,
  onChange,
  disabled,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}) {
  return (
    <div
      role="tablist"
      aria-label="Input source"
      className="border-edge inline-flex rounded-full border p-1"
    >
      {(["camera", "upload"] as const).map((value) => (
        <button
          key={value}
          role="tab"
          aria-selected={mode === value}
          disabled={disabled}
          onClick={() => onChange(value)}
          className={`rounded-full px-6 py-2 text-xs tracking-[0.18em] uppercase transition-colors duration-200 disabled:opacity-40 ${
            mode === value
              ? "bg-chalk text-ink"
              : "text-muted hover:text-chalk cursor-pointer"
          }`}
        >
          {value}
        </button>
      ))}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "secondary",
  disabled,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    primary: "bg-chalk text-ink hover:bg-white",
    secondary: "border border-edge text-chalk hover:border-faint",
    ghost: "text-muted hover:text-chalk",
  }[variant];

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-full px-6 py-2.5 text-xs tracking-[0.18em] uppercase transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${
        disabled ? "" : "cursor-pointer"
      }`}
    >
      {children}
    </button>
  );
}

export type IndicatorTone = "signal" | "warn" | "alert" | "idle";

export function Indicator({
  tone,
  label,
  pulse,
}: {
  tone: IndicatorTone;
  label: string;
  pulse?: boolean;
}) {
  const colour = {
    signal: "bg-signal",
    warn: "bg-warn",
    alert: "bg-alert",
    idle: "bg-faint",
  }[tone];

  return (
    <p className="text-muted flex items-center justify-center gap-2.5 text-xs tracking-[0.2em] uppercase">
      <span
        className={`h-1.5 w-1.5 rounded-full ${colour} ${pulse ? "animate-breathe" : ""}`}
      />
      {label}
    </p>
  );
}
