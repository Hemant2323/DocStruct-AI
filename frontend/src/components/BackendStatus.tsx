/**
 * Live backend connectivity indicator.
 *
 * This is the Milestone 1 proof that the frontend and backend actually talk to each
 * other: it calls the real health and readiness endpoints and reports what came back,
 * including failures.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { getHealth, getReadiness } from "@/api/health";
import type { HealthResponse, ReadinessResponse } from "@/types/api";

type ConnectionState =
  | { kind: "loading" }
  | { kind: "connected"; health: HealthResponse; readiness: ReadinessResponse }
  | { kind: "error"; message: string; code: string; requestId: string | null };

export function BackendStatus(): React.JSX.Element {
  const [state, setState] = useState<ConnectionState>({ kind: "loading" });

  const check = useCallback(async (): Promise<void> => {
    setState({ kind: "loading" });
    try {
      // Both probes are independent, so run them concurrently.
      const [health, readiness] = await Promise.all([
        getHealth(),
        getReadiness(),
      ]);
      setState({ kind: "connected", health, readiness });
    } catch (error) {
      // Surface the real reason instead of a generic "something went wrong".
      if (error instanceof ApiError) {
        setState({
          kind: "error",
          message: error.message,
          code: error.code,
          requestId: error.requestId,
        });
        return;
      }
      setState({
        kind: "error",
        message: error instanceof Error ? error.message : String(error),
        code: "unexpected_error",
        requestId: null,
      });
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Backend connection
        </h2>
        <button
          type="button"
          onClick={() => void check()}
          disabled={state.kind === "loading"}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          {state.kind === "loading" ? "Checking…" : "Re-check"}
        </button>
      </div>

      {state.kind === "loading" && (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Contacting the API…
        </p>
      )}

      {state.kind === "error" && (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
          <p className="flex items-center gap-2 font-medium text-red-800 dark:text-red-200">
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 rounded-full bg-red-500"
            />
            Not connected
          </p>
          <p className="mt-2 text-sm text-red-700 dark:text-red-300">
            {state.message}
          </p>
          <p className="mt-2 font-mono text-xs text-red-600 dark:text-red-400">
            {state.code}
            {state.requestId !== null && <> · {state.requestId}</>}
          </p>
          <p className="mt-3 text-sm text-red-700 dark:text-red-300">
            Start the backend with{" "}
            <code className="rounded bg-red-100 px-1 py-0.5 font-mono text-xs dark:bg-red-900">
              uvicorn app.main:app --reload
            </code>{" "}
            from the <code className="font-mono text-xs">backend/</code>{" "}
            directory.
          </p>
        </div>
      )}

      {state.kind === "connected" && (
        <div className="space-y-4">
          <p className="flex items-center gap-2 font-medium text-emerald-700 dark:text-emerald-300">
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500"
            />
            Connected
          </p>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
            <Field label="Application" value={state.health.app} />
            <Field label="Version" value={state.health.version} />
            <Field label="Environment" value={state.health.environment} />
          </dl>

          <div>
            <h3 className="mb-2 text-sm font-medium text-slate-700 dark:text-slate-300">
              Dependency checks
            </h3>
            <ul className="space-y-1">
              {Object.entries(state.readiness.checks).map(([name, healthy]) => (
                <li key={name} className="flex items-center gap-2 text-sm">
                  <span
                    aria-hidden="true"
                    className={`inline-block h-2 w-2 rounded-full ${
                      healthy ? "bg-emerald-500" : "bg-red-500"
                    }`}
                  />
                  <span className="text-slate-700 dark:text-slate-300">
                    {name}
                  </span>
                  <span
                    className={
                      healthy
                        ? "text-emerald-700 dark:text-emerald-400"
                        : "text-red-700 dark:text-red-400"
                    }
                  >
                    {healthy ? "healthy" : "unavailable"}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}

function Field({
  label,
  value,
}: {
  label: string;
  value: string;
}): React.JSX.Element {
  return (
    <div>
      <dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="font-medium text-slate-900 dark:text-slate-100">
        {value}
      </dd>
    </div>
  );
}
