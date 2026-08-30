/**
 * A small filled circle carrying one piece of state.
 *
 * Used where a word would be redundant — marking which tabs already hold a
 * result, or which files in an upload batch are still going. It replaced a "✓"
 * glyph that read as decoration and sat inconsistently against the label's
 * baseline.
 *
 * Always paired with visible text or an accessible label; the colour is never
 * the only thing carrying the meaning.
 */

type DotTone = "positive" | "caution" | "critical" | "muted";

const TONES: Record<DotTone, string> = {
  positive: "bg-positive",
  caution: "bg-caution",
  critical: "bg-critical",
  muted: "bg-ink-subtle",
};

interface StatusDotProps {
  tone?: DotTone;
  /**
   * Marks work still in progress. The pulse is the only thing distinguishing
   * "waiting" from "finished and neutral", so it carries information rather
   * than decorating — and it stops under `prefers-reduced-motion`, where the
   * accompanying text still says so.
   */
  pulse?: boolean;
  /** Announced to screen readers, which cannot see the colour. */
  label?: string;
}

export function StatusDot({
  tone = "muted",
  pulse = false,
  label,
}: StatusDotProps): React.JSX.Element {
  return (
    <span
      aria-hidden={label === undefined}
      aria-label={label}
      role={label === undefined ? undefined : "img"}
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${TONES[tone]} ${
        pulse ? "animate-pulse" : ""
      }`}
    />
  );
}
