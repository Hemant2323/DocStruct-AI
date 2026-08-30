/**
 * Changes & corrections: everything the AI proposed, and what became of it.
 *
 * Three groups, and the difference between them is the whole point of the
 * feature. **Applied** changes are in the document. **Suggestions** are not —
 * the document still shows the original value, and the change is here for a
 * person to decide on. **Refused** changes were judged unsafe and were never
 * offered; they are shown rather than discarded, because "the AI tried to change
 * this and was refused" is something a reviewer of a financial document needs to
 * be able to see.
 *
 * Every group is rendered the same way and none is collapsed by default, so a
 * refusal is exactly as visible as an applied change.
 */

import { EmptyState } from "@/components/ui/EmptyState";
import { StatusDot } from "@/components/ui/StatusDot";
import { confidenceLabel, correctionKindLabel } from "@/lib/conversion";
import type { ConversionCorrection, ConversionRejection } from "@/types/api";

/** The shape both applied changes and refusals reduce to for display. */
interface AuditEntry {
  key: string;
  kind: string;
  recordIndex: number;
  field: string;
  original: string;
  corrected: string;
  confidence: string;
  /** Why it was made, or why it was refused. */
  reason: string;
}

type GroupTone = "positive" | "caution" | "critical";

interface ConversionAuditProps {
  applied: ConversionCorrection[];
  suggestions: ConversionCorrection[];
  rejected: ConversionRejection[];
}

export function ConversionAudit({
  applied,
  suggestions,
  rejected,
}: ConversionAuditProps): React.JSX.Element {
  const total = applied.length + suggestions.length + rejected.length;

  return (
    <div className="border-line mt-6 border-t pt-5">
      {/* The three numbers stay on the heading. Unlike a count of traced
          fields, each one is a claim about what was done to the reader's
          document, and the difference between them is the entire feature. */}
      <h3 className="text-ink text-sm font-semibold">
        Changes &amp; corrections
        <span className="text-ink-subtle ml-2 font-normal tabular-nums">
          {applied.length} applied · {suggestions.length} suggested ·{" "}
          {rejected.length} refused
        </span>
      </h3>
      <p className="text-ink-muted mt-1 text-xs leading-relaxed">
        Only changes the system could prove were applied. Everything else is listed with
        the original value left standing.
      </p>

      {total === 0 ? (
        <div className="mt-3">
          <EmptyState>
            Nothing was changed. Every value in this document is your source text exactly
            as it was written.
          </EmptyState>
        </div>
      ) : (
        <div className="mt-3 space-y-4">
          <AuditGroup
            title="Applied"
            tone="positive"
            description="These changes are in the document above."
            entries={applied.map(fromCorrection)}
          />
          <AuditGroup
            title="Suggested, not applied"
            tone="caution"
            description="The document still shows the original value for each of these."
            entries={suggestions.map(fromCorrection)}
          />
          <AuditGroup
            title="Refused"
            tone="critical"
            description="Proposed changes the system judged unsafe and would not make."
            entries={rejected.map(fromRejection)}
          />
        </div>
      )}
    </div>
  );
}

function fromCorrection(item: ConversionCorrection): AuditEntry {
  return {
    key: item.id,
    kind: item.kind,
    recordIndex: item.record_index,
    field: item.field,
    original: item.original,
    corrected: item.corrected,
    confidence: item.confidence,
    reason: item.reason,
  };
}

function fromRejection(item: ConversionRejection, index: number): AuditEntry {
  return {
    // Refusals carry no id — they never became corrections — so position within
    // the refused list is the only stable identity available.
    key: `rejected-${index}`,
    kind: item.kind,
    recordIndex: item.record_index,
    field: item.field,
    original: item.original,
    corrected: item.corrected,
    confidence: item.confidence,
    reason: item.rejection_reason,
  };
}

function AuditGroup({
  title,
  tone,
  description,
  entries,
}: {
  title: string;
  tone: GroupTone;
  description: string;
  entries: AuditEntry[];
}): React.JSX.Element | null {
  // An empty group is omitted rather than shown as "0": three permanent empty
  // headings would bury the one group that has something in it.
  if (entries.length === 0) return null;

  return (
    <div>
      <h4 className="text-ink flex items-center gap-1.5 text-xs font-medium">
        <StatusDot tone={tone} />
        {title}
        <span className="text-ink-subtle font-normal tabular-nums">
          ({entries.length})
        </span>
      </h4>
      <p className="text-ink-subtle mt-0.5 mb-1.5 pl-3 text-xs">{description}</p>

      <ul className="divide-line border-line rounded-container divide-y border">
        {entries.map((entry) => (
          <li key={entry.key} className="px-2.5 py-2 text-xs">
            <p className="text-ink-subtle flex flex-wrap items-center gap-x-1.5">
              <span className="text-ink-muted font-mono">{entry.field}</span>
              <span aria-hidden="true">·</span>
              <span>record {entry.recordIndex + 1}</span>
              <span aria-hidden="true">·</span>
              <span>{correctionKindLabel(entry.kind)}</span>
              <span aria-hidden="true">·</span>
              <span>{confidenceLabel(entry.confidence)}</span>
            </p>

            {/* The two values side by side, because the change *is* the pair.
                Both are quoted so leading or trailing spaces — the entire point
                of a whitespace correction — stay visible. */}
            <p className="mt-1 flex flex-wrap items-baseline gap-1.5 wrap-anywhere">
              <span className="text-ink-muted line-through">“{entry.original}”</span>
              <span aria-hidden="true" className="text-ink-subtle">
                →
              </span>
              <span className={tone === "positive" ? "text-ink" : "text-ink-muted"}>
                “{entry.corrected}”
              </span>
            </p>

            {entry.reason !== "" && (
              <p className="text-ink-subtle mt-1">{entry.reason}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
