/**
 * The state Summary, Explain and Extract share.
 *
 * Held once, above the three panels, because it genuinely is shared: a single
 * `listInsights` call on open pre-fills all three tabs, and the context preview
 * and AI status are the same question whichever tab is showing. Giving each
 * panel its own copy would triple those requests on every document.
 *
 * "Optimize context" lives here for the same reason — it is a property of the
 * request, not of the tab, and switching tabs should not silently reset it.
 */

import { useCallback, useEffect, useState } from "react";

import {
  generateExplanation,
  generateExtraction,
  generateSummary,
  getAIStatus,
  getContextPreview,
  listInsights,
} from "@/api/analysis";
import { ApiError } from "@/api/client";
import type { AIStatus, ContextPreview, Insight, InsightKind } from "@/types/api";

export type RunState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string; requestId: string | null };

export interface InsightsController {
  results: Partial<Record<InsightKind, Insight>>;
  state: RunState;
  preview: ContextPreview | null;
  optimize: boolean;
  setOptimize: (value: boolean) => void;
  /** True once the server has said it cannot make AI requests at all. */
  aiUnavailable: boolean;
  run: (kind: InsightKind, refresh?: boolean) => Promise<void>;
}

export function useInsights(documentId: string): InsightsController {
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

  return {
    results,
    state,
    preview,
    optimize,
    setOptimize,
    aiUnavailable: status !== null && !status.configured,
    run,
  };
}
