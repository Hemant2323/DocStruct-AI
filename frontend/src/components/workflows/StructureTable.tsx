/**
 * The extracted records as a table: one row per record, one column per field.
 *
 * Columns come from the schema rather than from the names present in the
 * records, so every row lines up even when a record does not state a given
 * field — that cell is blank rather than the next value sliding into it.
 *
 * Sources stay reachable without crowding the table: a row expands to show its
 * own citations.
 */

import { Fragment, useState } from "react";

import { CitationList } from "@/components/CitationList";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { humanizeFieldName } from "@/lib/structureFields";
import type { Citation, StructureRecord } from "@/types/api";

interface StructureTableProps {
  records: StructureRecord[];
  columns: string[];
}

export function StructureTable({
  records,
  columns,
}: StructureTableProps): React.JSX.Element {
  const [openRecord, setOpenRecord] = useState<number | null>(null);

  if (records.length === 0) {
    return <EmptyState>No matching records were found in this document.</EmptyState>;
  }

  return (
    // The table scrolls inside its own container; a payment report can have many
    // columns, and the page itself must never scroll sideways. Focusable
    // because a region that only scrolls is otherwise unreachable without a
    // pointer — the columns past the right edge would simply not exist.
    <div
      role="region"
      aria-label="Extracted records"
      tabIndex={0}
      className="border-line rounded-container focus-visible:outline-accent overflow-x-auto border focus-visible:outline-2 focus-visible:-outline-offset-2"
    >
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-sunken border-line border-b">
            {columns.map((name) => (
              <th
                key={name}
                scope="col"
                className="text-ink-muted px-3 py-2 text-left text-xs font-medium whitespace-nowrap"
              >
                {humanizeFieldName(name)}
              </th>
            ))}
            <th scope="col" className="px-3 py-2 text-right">
              <span className="sr-only">Sources</span>
            </th>
          </tr>
        </thead>
        <tbody className="divide-line divide-y">
          {records.map((record, index) => {
            const byName = new Map(record.map((field) => [field.name, field]));
            const citations = record
              .map((field) => field.citation)
              .filter((citation): citation is Citation => citation !== null);
            const isOpen = openRecord === index;

            return (
              <Fragment key={index}>
                <tr className="align-top">
                  {columns.map((name) => (
                    <td
                      key={name}
                      className="text-ink px-3 py-2 whitespace-nowrap tabular-nums"
                    >
                      {byName.get(name)?.value ?? (
                        <span
                          title="This record does not state a value for this field."
                          className="text-ink-subtle"
                        >
                          —
                        </span>
                      )}
                    </td>
                  ))}
                  <td className="px-3 py-2 text-right">
                    <Button
                      variant="subtle"
                      size="sm"
                      onClick={() => setOpenRecord(isOpen ? null : index)}
                      aria-expanded={isOpen}
                      title="Show where this record's values came from"
                    >
                      {isOpen ? "Hide sources" : `Sources (${citations.length})`}
                    </Button>
                  </td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={columns.length + 1} className="px-3 pb-3">
                      {citations.length > 0 ? (
                        <CitationList citations={citations} />
                      ) : (
                        <p className="text-caution pt-2 text-xs">
                          None of this record’s values could be traced to a confirmed
                          location in your document.
                        </p>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
