/** A badge showing a document's processing state in plain language. */

import { statusCopy, type StatusTone } from "@/lib/documentStatus";
import type { ProcessingStatus } from "@/types/api";

const TONE_STYLES: Record<StatusTone, string> = {
  neutral: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200",
  info: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  success:
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  danger: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

const DOT_STYLES: Record<StatusTone, string> = {
  neutral: "bg-slate-400",
  info: "bg-blue-500",
  success: "bg-emerald-500",
  danger: "bg-red-500",
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
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${TONE_STYLES[copy.tone]}`}
    >
      <span
        aria-hidden="true"
        className={`inline-block h-1.5 w-1.5 rounded-full ${DOT_STYLES[copy.tone]} ${
          // A quiet pulse signals "still working" without an extra spinner.
          copy.inProgress ? "animate-pulse" : ""
        }`}
      />
      {copy.label}
    </span>
  );
}
