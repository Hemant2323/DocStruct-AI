/**
 * How long something has been running.
 *
 * Both waits in this product are long enough to need one: extraction runs in the
 * background after upload, and an AI request can take most of a minute. A
 * spinner with no counter gives no way to tell "slow" from "stuck", which is the
 * question a person actually has while waiting.
 *
 * One ticker per list rather than one per row — the callers pass `now` down and
 * subtract, so a library of fifty documents still runs a single interval.
 */

import { useEffect, useState } from "react";

/**
 * `8s`, `1m 04s`, `12m 30s`.
 *
 * Seconds are zero-padded past a minute so the number does not jump width as it
 * counts, which in a list would make every row twitch once a second.
 */
export function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  if (whole < 60) return `${whole}s`;

  const minutes = Math.floor(whole / 60);
  const remainder = whole % 60;
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

/**
 * The current time, re-read once a second while `active`.
 *
 * Stops entirely when nothing is in flight: a timer that keeps firing to
 * re-render an idle list is the kind of thing that quietly drains a laptop.
 */
export function useTickingNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return;

    // Re-read immediately: when the ticker restarts, the value it stopped at is
    // stale by however long it was idle.
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);

  return now;
}

/**
 * Seconds between an ISO timestamp and `now`, or `null` if unparseable.
 *
 * Clamped at zero. The timestamp comes from the server and `now` from the
 * browser, so a clock a few seconds apart would otherwise show a document that
 * has been processing for "-3s".
 */
export function secondsSince(isoTimestamp: string, now: number): number | null {
  const started = new Date(isoTimestamp).getTime();
  if (Number.isNaN(started)) return null;
  return Math.max(0, Math.floor((now - started) / 1000));
}
