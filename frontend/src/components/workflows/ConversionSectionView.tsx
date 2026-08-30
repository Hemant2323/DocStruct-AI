/**
 * One section of a converted document.
 *
 * Three shapes, because the server plans three: a `table` (a grid, one row per
 * record), a `detail` (one block per record, for few records with many fields)
 * and a `note` (prose the user asked for, holding no records). Each is rendered
 * as what it is rather than forced into a common layout — a fifteen-column
 * table squeezed into detail blocks, or three fields spread across a grid, are
 * both harder to read than the thing the plan already chose.
 *
 * What never varies is the treatment of a value: corrected text is marked,
 * absent text is blank rather than guessed, and a source label appears only
 * when the server confirmed it.
 */

import { Fragment, useState } from "react";

import { Button } from "@/components/ui/Button";
import { StatusDot } from "@/components/ui/StatusDot";
import type { ConvertedField, ConvertedRecord, ConvertedSection } from "@/types/api";

interface ConversionSectionViewProps {
  section: ConvertedSection;
  /** `name` → display label, from the plan. Falls back to the field's own label. */
  labels: Record<string, string>;
}

export function ConversionSectionView({
  section,
  labels,
}: ConversionSectionViewProps): React.JSX.Element {
  return (
    <section className="border-line border-t pt-3">
      <h4 className="text-ink mb-2 text-sm font-semibold">{section.heading}</h4>

      {section.kind === "note" && (
        <p className="text-ink-muted text-sm leading-relaxed wrap-anywhere whitespace-pre-wrap">
          {section.note}
        </p>
      )}

      {section.kind === "table" && (
        <TableSection section={section} labels={labels} />
      )}

      {section.kind === "detail" && (
        <DetailSection section={section} labels={labels} />
      )}

      {section.kind !== "note" && section.records.length === 0 && (
        <p className="text-ink-subtle text-xs">
          No records fell into this section.
        </p>
      )}
    </section>
  );
}

/** Resolve a field's display label: the plan's, then the field's own, then the key. */
function labelFor(
  name: string,
  labels: Record<string, string>,
  field: ConvertedField | undefined,
): string {
  return labels[name] ?? field?.label ?? name;
}

/** A grid: one row per record, one column per planned field. */
function TableSection({
  section,
  labels,
}: ConversionSectionViewProps): React.JSX.Element | null {
  const [openRecord, setOpenRecord] = useState<number | null>(null);

  if (section.records.length === 0) return null;

  // Columns come from the plan, not from the fields each record happens to
  // carry, so every row lines up and a field a record does not state leaves a
  // blank cell rather than shifting the next value into its place.
  const columns = section.columns;

  return (
    // Focusable: a converted report can be far wider than a phone, and a region
    // that only scrolls horizontally is unusable without a pointer otherwise.
    <div
      role="region"
      aria-label={`${section.heading} records`}
      tabIndex={0}
      className="border-line rounded-container focus-visible:outline-accent overflow-x-auto border focus-visible:outline-2 focus-visible:-outline-offset-2"
    >
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-sunken border-line border-b">
            {columns.map((name) => {
              const sample = section.records[0]?.fields.find(
                (field) => field.name === name,
              );
              return (
                <th
                  key={name}
                  scope="col"
                  className="text-ink-muted px-3 py-2 text-left text-xs font-medium whitespace-nowrap"
                >
                  {labelFor(name, labels, sample)}
                </th>
              );
            })}
            <th scope="col" className="px-3 py-2 text-right">
              <span className="sr-only">Sources</span>
            </th>
          </tr>
        </thead>
        <tbody className="divide-line divide-y">
          {section.records.map((record, index) => {
            const byName = new Map(record.fields.map((field) => [field.name, field]));
            const isOpen = openRecord === index;

            return (
              <Fragment key={`${record.source_record_index}-${index}`}>
                <tr className="align-top">
                  {columns.map((name) => (
                    <td key={name} className="text-ink px-3 py-2 align-top whitespace-nowrap">
                      <ValueText field={byName.get(name)} compact />
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
                      {/* Not "Sources (7)". The count of traced fields is a
                          measurement of the extraction, repeated on every row;
                          what the button does is open the list, where each
                          value says for itself whether it was traced. */}
                      {isOpen ? "Hide sources" : "Sources"}
                    </Button>
                  </td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={columns.length + 1} className="px-3 pb-3">
                      <RecordProvenance record={record} labels={labels} />
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

/** One block per record, each field on its own line. */
function DetailSection({
  section,
  labels,
}: ConversionSectionViewProps): React.JSX.Element | null {
  if (section.records.length === 0) return null;

  return (
    <ul className="divide-line border-line rounded-container divide-y border">
      {section.records.map((record, index) => (
        <li key={`${record.source_record_index}-${index}`} className="px-3 py-2.5">
          <dl className="space-y-1">
            {(section.columns.length > 0
              ? section.columns
              : record.fields.map((field) => field.name)
            ).map((name) => {
              const field = record.fields.find((item) => item.name === name);
              const source =
                field?.value.is_verified === true ? field.value.source_label : null;

              // A value the source wrote across several lines gets the label above it
              // and the text in a block of its own. Kept on one row it would be
              // squeezed into a narrow column and, worse, HTML would collapse its
              // indentation — which for source code is most of the meaning.
              if (isMultiline(field?.value.text)) {
                return (
                  <div key={name} className="pt-1 text-sm">
                    <dt className="text-ink-subtle flex items-baseline gap-2 text-xs">
                      {labelFor(name, labels, field)}
                      {source !== null && (
                        <span className="text-positive font-mono">{source}</span>
                      )}
                    </dt>
                    <dd className="mt-1">
                      <MultilineValue field={field} />
                    </dd>
                  </div>
                );
              }

              return (
                <div key={name} className="flex flex-wrap items-baseline gap-x-2 text-sm">
                  <dt className="text-ink-subtle min-w-32 text-xs">
                    {labelFor(name, labels, field)}
                  </dt>
                  <dd className="text-ink min-w-0 flex-1">
                    <ValueText field={field} />
                  </dd>
                  {source !== null && (
                    <dd className="text-positive shrink-0 font-mono text-xs">{source}</dd>
                  )}
                </div>
              );
            })}
          </dl>
        </li>
      ))}
    </ul>
  );
}

/**
 * Whether the source gave this value more than one line.
 *
 * A fact about the extracted text, not a guess about what kind of thing it is —
 * the same question the server's renderer asks, so the preview and the download
 * agree about which values are blocks. No language detection, no code sniffing.
 */
function isMultiline(text: string | undefined): boolean {
  return text !== undefined && (text.includes("\n") || text.includes("\r"));
}

/**
 * A value that spans lines, shown with its lines intact.
 *
 * `whitespace-pre` rather than `pre-wrap`: wrapping a long line of code would put
 * a false line break in the middle of a statement, so it scrolls instead — and
 * the region is focusable so it can be scrolled without a pointer.
 *
 * The change marker sits outside the block. Writing a glyph into the text would
 * alter the very content this component exists to preserve.
 */
function MultilineValue({
  field,
}: {
  field: ConvertedField | undefined;
}): React.JSX.Element | null {
  const value = field?.value;
  if (value === undefined) return null;

  return (
    <div>
      {value.origin === "corrected" && (
        <p className="text-caution mb-1 flex items-center gap-1.5 text-xs">
          <StatusDot tone="caution" />
          Corrected
          {value.original_text !== null && " — the original is in the audit below."}
        </p>
      )}
      <pre
        role="region"
        aria-label={field?.label ?? "value"}
        tabIndex={0}
        className="border-line bg-sunken rounded-container text-ink focus-visible:outline-accent overflow-x-auto border px-2.5 py-2 font-mono text-xs whitespace-pre focus-visible:outline-2 focus-visible:-outline-offset-2"
      >
        {value.text}
      </pre>
    </div>
  );
}

/**
 * One value.
 *
 * A missing field and a field the source does not state are the same thing to a
 * reader — neither is a value — so both render as a muted dash that says so on
 * hover. A corrected value carries a marker and keeps the original in its
 * tooltip; the tooltip is a convenience, and the full justification is in the
 * audit below the document.
 */
function ValueText({
  field,
  compact = false,
}: {
  field: ConvertedField | undefined;
  /** Set inside a table cell, where a row must stay one line tall. */
  compact?: boolean;
}): React.JSX.Element {
  const value = field?.value;

  if (value === undefined || value.origin === "absent" || value.text === "") {
    return (
      <span
        title="The source does not state this field for this record."
        className="text-ink-subtle"
      >
        —
      </span>
    );
  }

  // A table is a grid, and a forty-line value does not belong in one. Rather than
  // joining the lines with spaces — which reads as though that were the real value —
  // the first line is shown and the rest are counted, with the whole text on hover.
  // The complete value is always available in the Markdown and the downloads.
  if (compact && isMultiline(value.text)) {
    const lines = value.text.split(/\r\n|\r|\n/);
    const remaining = lines.length - 1;
    return (
      <span title={value.text} className="inline-flex items-baseline gap-1.5">
        <span className="font-mono">{lines[0]}</span>
        <span className="text-ink-subtle text-xs tabular-nums">
          +{remaining} more {remaining === 1 ? "line" : "lines"}
        </span>
      </span>
    );
  }

  if (value.origin === "corrected") {
    return (
      <span
        title={
          value.original_text !== null
            ? `Corrected from “${value.original_text}”. See Changes & corrections below.`
            : "Corrected. See Changes & corrections below."
        }
        className="inline-flex items-center gap-1.5"
      >
        <StatusDot tone="caution" label="corrected value" />
        <span className="text-ink">{value.text}</span>
      </span>
    );
  }

  return <span>{value.text}</span>;
}

/**
 * Where one record's values came from.
 *
 * Only confirmed locations appear. A value the server could not trace is listed
 * as unconfirmed rather than given a plausible-looking page number — the same
 * rule the rest of the product follows, and the reason a location shown here can
 * be trusted at all.
 */
function RecordProvenance({
  record,
  labels,
}: {
  record: ConvertedRecord;
  labels: Record<string, string>;
}): React.JSX.Element {
  const factual = record.fields.filter((field) => field.value.origin !== "structural");

  if (factual.length === 0) {
    return (
      <p className="text-ink-subtle pt-2 text-xs">
        This record holds no values taken from the source.
      </p>
    );
  }

  return (
    <div className="pt-2">
      <p className="text-ink-subtle mb-1.5 text-xs">
        From source record {record.source_record_index + 1}
      </p>
      <ul className="divide-line border-line rounded-container divide-y border">
        {factual.map((field) => (
          <li key={field.name} className="px-2.5 py-2 text-xs">
            <p className="flex flex-wrap items-center gap-1.5">
              <StatusDot tone={field.value.is_verified ? "positive" : "caution"} />
              <span className="text-ink-muted">{labelFor(field.name, labels, field)}</span>
              {field.value.is_verified && field.value.source_label !== null ? (
                <span className="text-positive font-mono font-medium">
                  {field.value.source_label}
                </span>
              ) : (
                <span
                  title="This value could not be matched to a specific place in your document. No location is shown rather than a guessed one."
                  className="text-caution"
                >
                  Source not verified
                </span>
              )}
            </p>
            {field.value.quoted_text !== null && field.value.quoted_text !== "" && (
              <p className="text-ink-muted mt-1 pl-3 italic wrap-anywhere">
                “{field.value.quoted_text}”
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
