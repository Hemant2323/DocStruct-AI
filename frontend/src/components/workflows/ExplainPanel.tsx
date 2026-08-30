/**
 * Explain: the same content, for someone new to the subject.
 *
 * Specialist terms are put in ordinary words, but conditions, deadlines and
 * figures stay exactly as written — dropping those is how a "simple"
 * explanation becomes a misleading one.
 */

import { InsightPanel } from "@/components/workflows/InsightPanel";
import type { InsightsController } from "@/components/workflows/useInsights";

export function ExplainPanel({
  insights,
}: {
  insights: InsightsController;
}): React.JSX.Element {
  return (
    <InsightPanel
      kind="explanation"
      action="Explain simply"
      hint="The same content, written for someone new to the subject."
      insights={insights}
    />
  );
}
