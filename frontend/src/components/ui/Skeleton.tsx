/**
 * A placeholder shaped like the result that is coming.
 *
 * These waits are long — a document conversion or a structured extraction can
 * take most of a minute — and a bare "Working…" gives no sense that anything is
 * happening. The shape also sets the expectation: prose lines for a summary,
 * rows for a table.
 *
 * The pulse is the one animation that carries information here, and it is
 * switched off with the rest under `prefers-reduced-motion`.
 */

import { useRef } from "react";

import { formatDuration, useTickingNow } from "@/lib/elapsed";

interface SkeletonProps {
  /** Roughly how many lines the result will occupy. */
  lines?: number;
  /** Announced while the placeholder is up. */
  label: string;
}

export function Skeleton({ lines = 3, label }: SkeletonProps): React.JSX.Element {
  return (
    <div role="status" aria-live="polite" className="space-y-2">
      <span className="sr-only">{label}</span>
      {Array.from({ length: lines }, (_, index) => (
        <div
          key={index}
          aria-hidden="true"
          // The last line is short, the way a paragraph's last line is.
          className={`bg-sunken h-3 animate-pulse rounded ${
            index === lines - 1 ? "w-2/5" : "w-full"
          }`}
        />
      ))}
    </div>
  );
}

/**
 * The wait itself, said plainly, under the placeholder — with a counter.
 *
 * The counter is what tells a slow request from a stuck one, and every AI
 * workflow already renders this note while it waits, so putting it here gives
 * all five of them the same treatment without any of them knowing about it.
 *
 * It appears only after a few seconds: a request that returns quickly should
 * not flash a stopwatch on its way past.
 */
const SHOW_ELAPSED_AFTER_SECONDS = 3;

export function LoadingNote({ children }: { children: React.ReactNode }): React.JSX.Element {
  // Mounts when the operation starts and unmounts when it ends, so the first
  // render is the start time.
  const startedAt = useRef(Date.now());
  const now = useTickingNow(true);
  const seconds = Math.max(0, Math.floor((now - startedAt.current) / 1000));

  return (
    <p className="text-ink-subtle mt-3 text-xs">
      {children}
      {seconds >= SHOW_ELAPSED_AFTER_SECONDS && (
        <span className="tabular-nums"> · {formatDuration(seconds)} elapsed</span>
      )}
    </p>
  );
}
