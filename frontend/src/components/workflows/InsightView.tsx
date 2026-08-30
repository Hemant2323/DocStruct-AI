/**
 * A generated insight: its prose, its extracted facts, and where they came from.
 *
 * Shared by Summary, Explain and Extract because the server returns one shape
 * for all three — prose in `content`, facts in `fields`, and never both.
 *
 * The distinction it exists to make visible is verified versus not. A fact whose
 * quote the server could not find in the stored document is shown *without* a
 * location rather than with a plausible one.
 */

import { CitationList } from "@/components/CitationList";
import { Notice } from "@/components/ui/Notice";
import { StatusDot } from "@/components/ui/StatusDot";
import type { Insight } from "@/types/api";

const UNVERIFIED_EXPLANATION =
  "The AI could not point to a specific place in your document for this value. " +
  "No location is shown rather than a guessed one.";

export function InsightView({ insight }: { insight: Insight }): React.JSX.Element {
  return (
    <div className="motion-safe:animate-rise">
      {insight.warnings.length > 0 && (
        <ul className="mb-3 space-y-1.5">
          {insight.warnings.map((warning) => (
            <li key={warning}>
              <Notice>{warning}</Notice>
            </li>
          ))}
        </ul>
      )}

      {insight.content !== "" && (
        <div className="text-ink text-sm leading-relaxed wrap-anywhere whitespace-pre-wrap">
          {insight.content}
        </div>
      )}

      {insight.fields.length > 0 && (
        <dl className="divide-line border-line rounded-container mt-3 divide-y border first:mt-0">
          {insight.fields.map((field) => {
            const verified = field.citation !== null && field.citation.is_verified;

            return (
              <div key={field.name} className="px-2.5 py-2">
                <dt className="text-ink-subtle font-mono text-xs">{field.name}</dt>
                <dd className="text-ink mt-0.5 text-sm font-medium wrap-anywhere">{field.value}</dd>
                {/* Same vocabulary as CitationList: a dot, then either the
                    resolved location or the plain admission that there isn't
                    one. The two must never look like different claims. */}
                <dd className="mt-1 flex items-center gap-1.5 text-xs">
                  <StatusDot tone={verified ? "positive" : "caution"} />
                  {verified && field.citation !== null ? (
                    <span className="text-positive font-mono">
                      {field.citation.location_label}
                    </span>
                  ) : (
                    <span title={UNVERIFIED_EXPLANATION} className="text-caution">
                      Source not verified
                    </span>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      )}

      <CitationList citations={insight.citations} />

      {/* Only what the reader chose or can act on. The model identifier was a
          build detail sitting where a footnote goes; the two flags left are
          answers to "why is this different from what I expected" — the context
          was trimmed, or this is the earlier result rather than a new one. */}
      {(insight.context_mode === "optimized" || insight.from_cache) && (
        <p className="text-ink-subtle mt-4 text-xs">
          {[
            insight.context_mode === "optimized" ? "Optimized context" : null,
            insight.from_cache ? "Saved result" : null,
          ]
            .filter((note) => note !== null)
            .join(" · ")}
        </p>
      )}
    </div>
  );
}
