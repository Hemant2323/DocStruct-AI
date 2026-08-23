/**
 * The AI panel for one document: summary, plain-language explanation, and extraction.
 *
 * Three things this component is careful about:
 *
 * - **Cost.** Each button is a paid request, so results already generated are loaded
 *   for free on open and shown without re-running anything.
 * - **Honesty.** Warnings from the server are displayed, not hidden, and a source is
 *   only ever shown as a location when the server confirmed it.
 * - **Not surprising the user.** "Optimize context for AI" is off by default, only
 *   offered on documents large enough for it to matter, and says in plain words that it
 *   does not touch their file.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import {
  generateExplanation,
  generateExtraction,
  generateSummary,
  getAIStatus,
  getContextPreview,
  listInsights,
} from "@/api/analysis";
import { CitationList } from "@/components/CitationList";
import type {
  AIStatus,
  ContextPreview,
  Insight,
  InsightKind,
} from "@/types/api";

const TABS: { kind: InsightKind; label: string; hint: string }[] = [
  {
    kind: "summary",
    label: "Summary",
    hint: "A short description of what this document says.",
  },
  {
    kind: "explanation",
    label: "Explain simply",
    hint: "The same content, written for someone new to the subject.",
  },
  {
    kind: "extraction",
    label: "Extract information",
    hint: "Key facts pulled out as structured data.",
  },
];

type RunState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string; requestId: string | null };

interface DocumentInsightsProps {
  documentId: string;
  filename: string;
}

export function DocumentInsights({
  documentId,
  filename,
}: DocumentInsightsProps): React.JSX.Element {
  const [active, setActive] = useState<InsightKind>("summary");
  const [results, setResults] = useState<Partial<Record<InsightKind, Insight>>>({});
  const [state, setState] = useState<RunState>({ kind: "idle" });
  const [optimize, setOptimize] = useState(false);
  const [preview, setPreview] = useState<ContextPreview | null>(null);
  const [status, setStatus] = useState<AIStatus | null>(null);

  // Load what already exists. Free — this never triggers an AI request.
  useEffect(() => {
    let cancelled = false;

    const load = async (): Promise<void> => {
      try {
        const [existing, contextPreview, aiStatus] = await Promise.all([
          listInsights(documentId),
          getContextPreview(documentId),
          getAIStatus(),
        ]);
        if (cancelled) return;

        const byKind: Partial<Record<InsightKind, Insight>> = {};
        for (const insight of existing.items) {
          byKind[insight.kind] = insight;
        }
        setResults(byKind);
        setPreview(contextPreview);
        setStatus(aiStatus);
      } catch {
        // Not fatal: the buttons still work, there is just nothing to pre-fill.
        if (!cancelled) setPreview(null);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const run = useCallback(
    async (kind: InsightKind, refresh = false): Promise<void> => {
      setActive(kind);
      setState({ kind: "loading" });

      const options = { optimize_context: optimize, refresh };

      try {
        const insight =
          kind === "summary"
            ? await generateSummary(documentId, options)
            : kind === "explanation"
              ? await generateExplanation(documentId, options)
              : await generateExtraction(documentId, options);

        setResults((previous) => ({ ...previous, [kind]: insight }));
        setState({ kind: "idle" });
      } catch (error) {
        setState({
          kind: "error",
          message:
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try again.",
          requestId: error instanceof ApiError ? error.requestId : null,
        });
      }
    },
    [documentId, optimize],
  );

  const current = results[active];
  const isLoading = state.kind === "loading";
  const aiUnavailable = status !== null && !status.configured;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          AI understanding
        </h2>
        <p className="mt-0.5 truncate text-sm text-slate-500 dark:text-slate-400">
          {filename}
        </p>
      </div>

      {aiUnavailable && (
        <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
          AI features are not configured on this server. An Anthropic API key needs to be
          set in the backend’s <code className="font-mono text-xs">.env</code> file.
        </div>
      )}

      {/* Tabs */}
      <div className="mb-4 flex flex-wrap gap-2">
        {TABS.map((tab) => (
          <button
            key={tab.kind}
            type="button"
            title={tab.hint}
            onClick={() => setActive(tab.kind)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
              active === tab.kind
                ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "border border-slate-300 text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
            }`}
          >
            {tab.label}
            {results[tab.kind] !== undefined && (
              <span className="ml-1.5 text-xs opacity-70">✓</span>
            )}
          </button>
        ))}
      </div>

      {/* Optional context optimization — only where it would actually help. */}
      {preview?.optimization_available === true && (
        <label className="mb-4 flex cursor-pointer items-start gap-2.5 rounded-md border border-slate-200 p-3 dark:border-slate-700">
          <input
            type="checkbox"
            checked={optimize}
            onChange={(event) => setOptimize(event.target.checked)}
            className="mt-0.5"
          />
          <span className="text-sm">
            <span className="font-medium text-slate-800 dark:text-slate-200">
              Optimize context for AI
            </span>
            <span className="mt-0.5 block text-xs text-slate-600 dark:text-slate-400">
              Reduce unnecessary context for large documents while preserving important
              information. This changes only what the AI is given — your uploaded file and
              its extracted content are not changed.
            </span>
          </span>
        </label>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={isLoading || aiUnavailable}
          onClick={() => void run(active)}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
        >
          {isLoading
            ? "Working…"
            : current !== undefined
              ? "Show again"
              : `Generate ${TABS.find((tab) => tab.kind === active)?.label.toLowerCase()}`}
        </button>

        {current !== undefined && (
          <button
            type="button"
            disabled={isLoading || aiUnavailable}
            onClick={() => void run(active, true)}
            title="Runs the AI again instead of showing the saved result."
            className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            Regenerate
          </button>
        )}

        {preview !== null && (
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {preview.chunk_count} section{preview.chunk_count === 1 ? "" : "s"} ·{" "}
            {preview.estimated_tokens.toLocaleString()} tokens
          </span>
        )}
      </div>

      {isLoading && (
        <div className="rounded-md border border-slate-200 p-6 text-center dark:border-slate-700">
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Reading the document…
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
            This can take up to a minute for a long document.
          </p>
        </div>
      )}

      {state.kind === "error" && (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
          <p className="text-sm text-red-800 dark:text-red-200">{state.message}</p>
          {state.requestId !== null && (
            <p className="mt-2 text-xs text-red-600 dark:text-red-400">
              Reference: <code className="font-mono">{state.requestId}</code>
            </p>
          )}
        </div>
      )}

      {!isLoading && state.kind !== "error" && current === undefined && (
        <p className="rounded-md border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600 dark:border-slate-600 dark:text-slate-400">
          {TABS.find((tab) => tab.kind === active)?.hint}
        </p>
      )}

      {!isLoading && current !== undefined && (
        <InsightView insight={current} />
      )}
    </section>
  );
}

function InsightView({ insight }: { insight: Insight }): React.JSX.Element {
  return (
    <div>
      {insight.warnings.length > 0 && (
        <ul className="mb-3 space-y-1">
          {insight.warnings.map((warning) => (
            <li
              key={warning}
              className="rounded-md border border-amber-200 bg-amber-50 p-2.5 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200"
            >
              {warning}
            </li>
          ))}
        </ul>
      )}

      {insight.content !== "" && (
        <div className="text-sm whitespace-pre-wrap text-slate-800 dark:text-slate-200">
          {insight.content}
        </div>
      )}

      {insight.fields.length > 0 && (
        <dl className="divide-y divide-slate-200 dark:divide-slate-700">
          {insight.fields.map((field) => (
            <div key={field.name} className="py-2.5">
              <dt className="font-mono text-xs text-slate-500 dark:text-slate-400">
                {field.name}
              </dt>
              <dd className="mt-0.5 text-sm font-medium text-slate-900 dark:text-slate-100">
                {field.value}
              </dd>
              <dd className="mt-1 text-xs">
                {field.citation !== null && field.citation.is_verified ? (
                  <span className="text-emerald-700 dark:text-emerald-400">
                    Source: {field.citation.location_label}
                  </span>
                ) : (
                  <span
                    title="The AI could not point to a specific place in your document for this value. No location is shown rather than a guessed one."
                    className="text-amber-700 dark:text-amber-400"
                  >
                    Source not verified
                  </span>
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}

      <CitationList citations={insight.citations} />

      <p className="mt-4 text-xs text-slate-500 dark:text-slate-500">
        {insight.model}
        {insight.context_mode === "optimized" && " · optimized context"}
        {insight.from_cache && " · saved result"}
      </p>
    </div>
  );
}
