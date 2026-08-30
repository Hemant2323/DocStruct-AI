/**
 * A document's processing state, in plain language.
 *
 * Neutral-first: `pending` and `processing` are both ordinary progress, so
 * neither gets a colour — a pulsing dot says work is happening and the label
 * says which stage. Colour is spent only where the outcome matters, on the two
 * terminal states. That way a library of ten documents mid-upload reads as a
 * quiet list rather than a wall of blue.
 */

import { StatusDot } from "@/components/ui/StatusDot";
import { statusCopy, type StatusTone } from "@/lib/documentStatus";
import type { ProcessingStatus } from "@/types/api";

const TONE_STYLES: Record<StatusTone, string> = {
  neutral: "border-line text-ink-muted",
  info: "border-line text-ink-muted",
  success: "border-positive/40 text-positive",
  danger: "border-critical/40 text-critical",
};

const DOT_TONES: Record<StatusTone, "muted" | "positive" | "critical"> = {
  neutral: "muted",
  info: "muted",
  success: "positive",
  danger: "critical",
};

interface DocumentStatusBadgeProps {
  status: ProcessingStatus;
}

export function DocumentStatusBadge({
  status,
}: DocumentStatusBadgeProps): React.JSX.Element {
  const copy = statusCopy(status);

  return (
    <span
      // The description is the accessible name: a screen reader hears the full
      // explanation, while sighted users see the short label plus a tooltip.
      title={copy.description}
      aria-label={`Status: ${copy.label}. ${copy.description}`}
      className={`rounded-control inline-flex shrink-0 items-center gap-1.5 border px-1.5 py-0.5 text-xs font-medium ${TONE_STYLES[copy.tone]}`}
    >
      <StatusDot tone={DOT_TONES[copy.tone]} pulse={copy.inProgress} />
      {copy.label}
    </span>
  );
}
