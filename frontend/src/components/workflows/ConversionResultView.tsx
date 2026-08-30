/**
 * A converted document, and everything that justifies it.
 *
 * Order matters here. The document comes first, because it is what was asked
 * for. The downloads sit directly under it, because taking it away is the next
 * thing anyone wants. The audit comes after both — far enough down not to
 * compete with the document, close enough that nobody can read the document
 * without meeting it.
 *
 * The plan and the raw Markdown are disclosures. Both are genuinely useful —
 * one explains why the document is shaped as it is, the other is exactly what
 * the `.md` download contains — but neither is what a reader came for.
 */

import { Button } from "@/components/ui/Button";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { Notice } from "@/components/ui/Notice";
import { ConversionAudit } from "@/components/workflows/ConversionAudit";
import { ConversionSectionView } from "@/components/workflows/ConversionSectionView";
import type { RunState } from "@/components/workflows/useInsights";
import {
  DOWNLOAD_FORMATS,
  MARKDOWN_EXPORT_FORMAT,
  visibleConversionWarnings,
} from "@/lib/conversion";
import { formatDate } from "@/lib/format";
import type { ConversionExportFormat, ConversionResponse } from "@/types/api";

interface ConversionResultViewProps {
  conversion: ConversionResponse;
  isExporting: boolean;
  exportingFormat: ConversionExportFormat | null;
  exportState: RunState;
  onExport: (format: ConversionExportFormat) => void;
}

export function ConversionResultView({
  conversion,
  isExporting,
  exportingFormat,
  exportState,
  onExport,
}: ConversionResultViewProps): React.JSX.Element {
  const labels = conversion.plan.labels;
  // The response's own `warnings` is untouched — this is only what gets a banner.
  const warnings = visibleConversionWarnings(conversion.warnings);

  return (
    <div className="motion-safe:animate-rise">
      {warnings.length > 0 && (
        <ul className="mb-4 space-y-1.5">
          {warnings.map((warning) => (
            <li key={warning}>
              <Notice>{warning}</Notice>
            </li>
          ))}
        </ul>
      )}

      {/* The masthead. The title is the largest thing on the page because the
          document is what was asked for; the instruction sits under it, set
          apart by a rule, because reading the two together is how anyone checks
          whether they got what they asked for. */}
      <header className="border-line mb-5 border-b pb-4">
        <h3 className="text-ink text-lg font-semibold tracking-tight wrap-anywhere">
          {conversion.title}
        </h3>
        <p className="text-ink-subtle mt-1 text-xs tabular-nums">
          {conversion.sections.length} section
          {conversion.sections.length === 1 ? "" : "s"} ·{" "}
          {conversion.source_record_count} source record
          {conversion.source_record_count === 1 ? "" : "s"} ·{" "}
          {formatDate(conversion.created_at)}
        </p>
        <p className="border-line text-ink-muted mt-2.5 border-l-2 pl-3 text-xs leading-relaxed italic wrap-anywhere">
          {conversion.instruction}
        </p>
      </header>

      <div className="space-y-6">
        {conversion.sections.map((section, index) => (
          <ConversionSectionView
            key={`${section.heading}-${index}`}
            section={section}
            labels={labels}
          />
        ))}
      </div>

      {conversion.sections.length === 0 && (
        <p className="text-ink-subtle text-sm">
          This conversion produced no sections. Nothing in the source matched what was
          asked for.
        </p>
      )}

      <div className="border-line mt-6 border-t pt-5">
        <h3 className="text-ink text-sm font-semibold">Download</h3>
        <p className="text-ink-muted mt-1 mb-3 text-xs leading-relaxed">
          Every format carries the audit below, so a tidied report never arrives without
          saying what was tidied.
        </p>

        {/* One cell each, from one list: same variant, same size, same width, and the
            grid gives them the same column. Equal weight has to be structural — two
            buttons styled alike still read as ranked if one is wider or set first in a
            row that wraps. */}
        <div className="grid gap-2 sm:grid-cols-2">
          {DOWNLOAD_FORMATS.map((format) => (
            <div
              key={format.value}
              className="border-line hover:border-line-strong rounded-container flex flex-col gap-2.5 border p-3.5 transition-colors duration-100"
            >
              <Button
                variant="secondary"
                disabled={isExporting}
                onClick={() => onExport(format.value)}
                className="w-full"
              >
                {isExporting && exportingFormat === format.value
                  ? "Preparing…"
                  : `Download ${format.label}`}
              </Button>
              <p className="text-ink-subtle text-xs leading-relaxed">{format.hint}</p>
            </div>
          ))}
        </div>

        {/* Markdown sits below the pair rather than in it — available, plainly not one
            of the two, and next to the source it corresponds to. */}
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button
            variant="subtle"
            size="sm"
            disabled={isExporting}
            title={MARKDOWN_EXPORT_FORMAT.hint}
            onClick={() => onExport(MARKDOWN_EXPORT_FORMAT.value)}
          >
            {isExporting && exportingFormat === MARKDOWN_EXPORT_FORMAT.value
              ? "Preparing…"
              : MARKDOWN_EXPORT_FORMAT.label}
          </Button>
          <span className="text-ink-subtle text-xs">{MARKDOWN_EXPORT_FORMAT.hint}</span>
        </div>

        {exportState.kind === "error" && (
          <div className="mt-3">
            <ErrorBlock message={exportState.message} requestId={exportState.requestId} />
          </div>
        )}
      </div>

      <ConversionAudit
        applied={conversion.corrections}
        suggestions={conversion.suggestions}
        rejected={conversion.rejected}
      />

      <details className="border-line rounded-container mt-6 border">
        <summary className="text-ink hover:bg-sunken focus-visible:outline-accent cursor-pointer rounded-t-[5px] px-3 py-2.5 text-sm font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2">
          How this was organised
        </summary>
        <div className="border-line space-y-2 border-t px-3 py-2.5">
          <p className="text-ink-muted text-xs">
            The AI chose the layout and proposed the corrections. The document itself was
            assembled from your data by ordinary code, so the same source and the same
            decisions always produce the same document.
          </p>

          <ul className="space-y-1">
            {conversion.plan.sections.map((section, index) => (
              <li key={`${section.heading}-${index}`} className="text-ink-muted text-xs">
                <span className="text-ink font-medium">{section.heading}</span> —{" "}
                {section.kind}
                {section.group_by !== null && ` · grouped by ${section.group_by}`}
                {section.columns.length > 0 && (
                  <span className="text-ink-subtle font-mono">
                    {" "}
                    {section.columns.join(", ")}
                  </span>
                )}
              </li>
            ))}
          </ul>

          {Object.keys(conversion.vocabularies).length > 0 && (
            <div>
              <p className="text-ink text-xs font-medium">Standardised value sets</p>
              <ul className="mt-0.5 space-y-0.5">
                {Object.entries(conversion.vocabularies).map(([field, values]) => (
                  <li key={field} className="text-ink-muted text-xs">
                    <span className="font-mono">{field}</span>:{" "}
                    <span className="text-ink-subtle">{values.join(", ")}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </details>

      <details className="border-line rounded-container mt-2 border">
        <summary className="text-ink hover:bg-sunken focus-visible:outline-accent cursor-pointer rounded-t-[5px] px-3 py-2.5 text-sm font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2">
          Markdown source
        </summary>
        <div className="border-line border-t px-3 py-2.5">
          <p className="text-ink-subtle mb-2 text-xs">
            Exactly what the Markdown download contains, returned from the stored
            rendering rather than rebuilt — so a download cannot disagree with this.
          </p>
          <pre
            role="region"
            aria-label="Markdown source"
            tabIndex={0}
            className="text-ink-muted focus-visible:outline-accent overflow-x-auto font-mono text-xs whitespace-pre focus-visible:outline-2 focus-visible:-outline-offset-2"
          >
            {conversion.markdown}
          </pre>
        </div>
      </details>

      {/* The model identifier used to sit here. It is a build detail — a reader
          cannot act on it, and the response still carries it for support. What
          is kept is the one fact that explains something the reader chose: the
          source was not read again, which is what the option above controls. */}
      {conversion.records_from_cache && (
        <p className="text-ink-subtle mt-5 text-xs">
          Built from the previous reading of this document, not a fresh one.
        </p>
      )}
    </div>
  );
}
