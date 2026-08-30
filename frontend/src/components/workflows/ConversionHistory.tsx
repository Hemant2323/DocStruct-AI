/**
 * Every conversion already produced for this document.
 *
 * Worth its own list because there is no conversion-level cache on the server:
 * the same instruction legitimately yields a different document, so each run is
 * kept as its own result. Re-opening one costs nothing and never re-runs the AI,
 * which makes this the cheap way back to a document you have already paid for.
 *
 * Rows carry their instruction and their audit counts, because those are what
 * tell two conversions of the same document apart.
 */

import { StatusDot } from "@/components/ui/StatusDot";
import { formatDate } from "@/lib/format";
import type { ConversionSummary } from "@/types/api";

interface ConversionHistoryProps {
  items: ConversionSummary[];
  /** The conversion currently on screen, if it is one of these. */
  activeId: string | null;
  /** True while one is being fetched, so rows can be disabled. */
  isLoading: boolean;
  onOpen: (conversionId: string) => void;
}

export function ConversionHistory({
  items,
  activeId,
  isLoading,
  onOpen,
}: ConversionHistoryProps): React.JSX.Element | null {
  if (items.length === 0) return null;

  return (
    <div className="border-line border-t pt-3">
      <h3 className="text-ink text-sm font-medium">
        Past conversions
        <span className="text-ink-subtle ml-2 font-normal tabular-nums">
          {items.length}
        </span>
      </h3>
      <p className="text-ink-muted mt-0.5 mb-2 text-xs">
        Opening one is free — the stored result is returned exactly as it was produced.
      </p>

      <ul className="divide-line border-line rounded-container divide-y border">
        {items.map((item) => {
          const isActive = item.id === activeId;

          return (
            <li key={item.id}>
              <button
                type="button"
                disabled={isLoading}
                aria-current={isActive ? "true" : undefined}
                onClick={() => onOpen(item.id)}
                className={`focus-visible:outline-accent w-full px-2.5 py-2 text-left transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${
                  isActive ? "bg-accent-soft" : "hover:bg-sunken"
                }`}
              >
                <span className="flex items-baseline gap-2">
                  <span className="text-ink min-w-0 flex-1 truncate text-sm font-medium">
                    {item.title}
                  </span>
                  <span className="text-ink-subtle shrink-0 text-xs tabular-nums">
                    {formatDate(item.created_at)}
                  </span>
                </span>

                <span className="text-ink-muted mt-0.5 block truncate text-xs italic">
                  “{item.instruction}”
                </span>

                <span className="text-ink-subtle mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs tabular-nums">
                  <span>
                    {item.source_record_count} record
                    {item.source_record_count === 1 ? "" : "s"}
                  </span>
                  {item.applied_correction_count > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <StatusDot tone="positive" />
                      {item.applied_correction_count} applied
                    </span>
                  )}
                  {item.suggestion_count > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <StatusDot tone="caution" />
                      {item.suggestion_count} suggested
                    </span>
                  )}
                  {item.rejected_correction_count > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <StatusDot tone="critical" />
                      {item.rejected_correction_count} refused
                    </span>
                  )}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
