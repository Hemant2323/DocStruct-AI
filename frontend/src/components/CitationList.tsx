/**
 * Shows where an AI result came from.
 *
 * The distinction this component exists to make visible is **verified versus
 * not**. A confirmed source shows the place in the file it came from; one that
 * could not be confirmed says so plainly. Nothing is ever displayed as a
 * location unless the server resolved it from stored content, so a user can
 * trust what they see here.
 *
 * Rows separated by hairlines rather than each quote sitting in its own filled
 * box: this list already renders inside a panel, and a stack of tinted cards
 * inside a card made four sources look like four separate results.
 */

import { StatusDot } from "@/components/ui/StatusDot";
import type { Citation } from "@/types/api";

const UNVERIFIED_EXPLANATION =
  "The AI referred to something we could not match to a specific place in your " +
  "document. No location is shown rather than a guessed one.";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps): React.JSX.Element | null {
  if (citations.length === 0) {
    return null;
  }

  return (
    <div className="border-line mt-4 border-t pt-3">
      {/* No "N of M confirmed" tally. A ratio of confirmed sources is a
          measurement of the system, and a large one read as a diagnostic
          rather than as information — while saying nothing about *which*
          source was unconfirmed. Each row below still states that outright,
          which is the part a reader can act on. */}
      <h3 className="text-ink mb-2 text-sm font-medium">Sources</h3>

      <ul className="divide-line border-line rounded-container divide-y border">
        {citations.map((citation, index) => (
          <li key={`${citation.location_label}-${index}`} className="px-2.5 py-2 text-xs">
            <p className="flex items-center gap-1.5">
              <StatusDot tone={citation.is_verified ? "positive" : "caution"} />
              {citation.is_verified ? (
                <span className="text-positive font-mono font-medium">
                  {citation.location_label}
                </span>
              ) : (
                <span title={UNVERIFIED_EXPLANATION} className="text-caution font-medium">
                  Source not verified
                </span>
              )}
            </p>
            <p className="text-ink-muted mt-1 pl-3 italic wrap-anywhere">“{citation.quoted_text}”</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
