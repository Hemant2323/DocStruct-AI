/**
 * The shared body of the Summary, Explain and Extract tabs.
 *
 * Those three differ only in wording and in which endpoint they call — the
 * cost warning, the saved-result handling, the regenerate affordance and the
 * context-optimization option are identical. Writing them out three times would
 * mean three places to fix the next time one of those behaviours changes, so
 * each tab is a thin configuration of this.
 */

import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { LoadingNote, Skeleton } from "@/components/ui/Skeleton";
import { InsightView } from "@/components/workflows/InsightView";
import type { InsightsController } from "@/components/workflows/useInsights";
import type { InsightKind } from "@/types/api";

interface InsightPanelProps {
  kind: InsightKind;
  /** Used in the button: "Generate summary". */
  action: string;
  /** What this tab produces, shown before anything has been generated. */
  hint: string;
  insights: InsightsController;
}

export function InsightPanel({
  kind,
  action,
  hint,
  insights,
}: InsightPanelProps): React.JSX.Element {
  const { results, state, preview, optimize, setOptimize, aiUnavailable, run } = insights;

  const current = results[kind];
  const isLoading = state.kind === "loading";

  return (
    <div className="space-y-4">
      {/* Offered only where it would actually help; on a small document,
          reducing context only loses detail. */}
      {preview?.optimization_available === true && (
        <label className="border-line rounded-container flex cursor-pointer items-start gap-2.5 border p-3">
          <input
            type="checkbox"
            checked={optimize}
            onChange={(event) => setOptimize(event.target.checked)}
            className="accent-accent mt-0.5"
          />
          <span className="text-sm">
            <span className="text-ink font-medium">Optimize context for AI</span>
            <span className="text-ink-muted mt-0.5 block text-xs">
              Reduce unnecessary context for large documents while preserving important
              information. This changes only what the AI is given — your uploaded file and
              its extracted content are not changed.
            </span>
          </span>
        </label>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          disabled={isLoading || aiUnavailable}
          onClick={() => void run(kind)}
        >
          {isLoading ? "Working…" : current !== undefined ? "Show again" : action}
        </Button>

        {current !== undefined && (
          <Button
            disabled={isLoading || aiUnavailable}
            onClick={() => void run(kind, true)}
            title="Runs the AI again instead of showing the saved result."
          >
            Regenerate
          </Button>
        )}

        {/* The section and token counts that stood here described the request
            being assembled, not the document. A token is a unit of the model's
            billing, and nothing on this panel changes either number, so it read
            as instrumentation. The preview is still fetched — the optimization
            option above depends on it. */}
      </div>

      {isLoading && (
        <div>
          <Skeleton lines={4} label="Reading the document" />
          <LoadingNote>
            Reading the document — this can take up to a minute for a long document.
          </LoadingNote>
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBlock message={state.message} requestId={state.requestId} />
      )}

      {!isLoading && state.kind !== "error" && current === undefined && (
        <EmptyState>{hint}</EmptyState>
      )}

      {!isLoading && current !== undefined && <InsightView insight={current} />}
    </div>
  );
}
