/**
 * Summary: a short description of what the document says.
 *
 * Grounded in the document only — the model is instructed to say when something
 * is unclear rather than fill it in from general knowledge.
 */

import { InsightPanel } from "@/components/workflows/InsightPanel";
import type { InsightsController } from "@/components/workflows/useInsights";

export function SummaryPanel({
  insights,
}: {
  insights: InsightsController;
}): React.JSX.Element {
  return (
    <InsightPanel
      kind="summary"
      action="Generate summary"
      hint="A short description of what this document says, cited back to it."
      insights={insights}
    />
  );
}
