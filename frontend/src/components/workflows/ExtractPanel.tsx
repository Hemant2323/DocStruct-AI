/**
 * Extract: key facts pulled out as structured data.
 *
 * Each fact carries the wording the model relied on, which the server then
 * searches for in the stored document. A fact whose quote cannot be found is
 * still returned, but explicitly without a location.
 */

import { InsightPanel } from "@/components/workflows/InsightPanel";
import type { InsightsController } from "@/components/workflows/useInsights";

export function ExtractPanel({
  insights,
}: {
  insights: InsightsController;
}): React.JSX.Element {
  return (
    <InsightPanel
      kind="extraction"
      action="Extract information"
      hint="Key facts pulled out as structured data, each with its source."
      insights={insights}
    />
  );
}
