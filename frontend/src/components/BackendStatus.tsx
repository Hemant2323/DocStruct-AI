/**
 * Live backend connectivity — shown only when there is something wrong.
 *
 * The probes are unchanged: both real endpoints, called on mount, with failures
 * reported rather than swallowed. What changed is that a healthy result now
 * renders nothing. "Backend connected" is a fact about our infrastructure, and
 * a person using the product has no decision that depends on it; a permanent
 * green light saying the software is working is the sort of thing an operator
 * needs and a user reads as clutter. The app working *is* the indicator.
 *
 * A failure is still not collapsed, and deliberately so. If the API is
 * unreachable then nothing else on the page will work, and the user is entitled
 * to know that before they blame their own file. What the banner no longer
 * carries is the internal error code and the shell command to start the server
 * — both are for whoever runs the backend, and both are still in the console
 * and the API response for them.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { getHealth, getReadiness } from "@/api/health";
import { StatusDot } from "@/components/ui/StatusDot";
import type { ReadinessResponse } from "@/types/api";

type ConnectionState =
  | { kind: "loading" }
  | { kind: "connected"; readiness: ReadinessResponse }
  | { kind: "error" };

export function BackendStatus(): React.JSX.Element | null {
  const [state, setState] = useState<ConnectionState>({ kind: "loading" });

  const check = useCallback(async (): Promise<void> => {
    setState({ kind: "loading" });
    try {
      // Both probes are independent, so run them concurrently. Health is still
      // called — an app that reports "reachable" without having reached it is
      // not a check — but its build metadata is no longer shown to anyone.
      const [, readiness] = await Promise.all([getHealth(), getReadiness()]);
      setState({ kind: "connected", readiness });
    } catch (error) {
      // The reason goes to the console, not the sidebar. It is what whoever
      // runs the backend needs and nothing a user can act on — but it is never
      // swallowed, because a connection failure with no recorded cause is the
      // hardest kind to diagnose later.
      const detail =
        error instanceof ApiError
          ? `${error.code}${error.requestId === null ? "" : ` · ${error.requestId}`}: ${error.message}`
          : error instanceof Error
            ? error.message
            : String(error);
      console.error(`Backend health check failed — ${detail}`);
      setState({ kind: "error" });
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const degraded =
    state.kind === "connected" &&
    Object.values(state.readiness.checks).some((healthy) => !healthy);

  // Working, or not yet known: say nothing. A check that has not come back is
  // not news either — announcing "contacting the API…" on every load made the
  // first second of the app about the app.
  if (state.kind === "loading" || (state.kind === "connected" && !degraded)) {
    return null;
  }

  return (
    <section
      aria-live="polite"
      className="border-line motion-safe:animate-rise border-t px-3 py-2.5 text-xs"
    >
      <p className="flex items-center gap-2">
        <StatusDot tone={state.kind === "error" ? "critical" : "caution"} />
        <span className="text-ink font-medium">
          {state.kind === "error" ? "Can’t reach the server" : "Some features are down"}
        </span>
        <button
          type="button"
          onClick={() => void check()}
          className="text-ink-subtle hover:text-ink focus-visible:outline-accent rounded-control ml-auto shrink-0 transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1"
        >
          Try again
        </button>
      </p>

      <p className="text-ink-muted mt-1.5 leading-relaxed">
        {state.kind === "error"
          ? "Uploads and AI features won’t work until the connection is back."
          : "Part of the service is unavailable, so some features may fail."}
      </p>
    </section>
  );
}
