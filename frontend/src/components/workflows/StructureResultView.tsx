/**
 * A generated structured document: the table, the schema behind it, and the
 * ways to take it away.
 *
 * The schema is shown as a disclosure rather than a form — when the fields were
 * worked out from a description, seeing what the server decided to extract is
 * how you check it understood you, but it is not something to edit here.
 */

import { Button } from "@/components/ui/Button";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { Notice } from "@/components/ui/Notice";
import { StructureTable } from "@/components/workflows/StructureTable";
import type { RunState } from "@/components/workflows/useInsights";
import { humanizeFieldName, STRUCTURE_EXPORT_FORMATS } from "@/lib/structureFields";
import type { StructureExportFormat, StructureResponse } from "@/types/api";

interface StructureResultViewProps {
  result: StructureResponse;
  aiUnavailable: boolean;
  isExporting: boolean;
  exportingFormat: StructureExportFormat | null;
  exportState: RunState;
  onExport: (format: StructureExportFormat) => void;
}

export function StructureResultView({
  result,
  aiUnavailable,
  isExporting,
  exportingFormat,
  exportState,
  onExport,
}: StructureResultViewProps): React.JSX.Element {
  const columns = Object.keys(result.schema_used);

  return (
    <div className="motion-safe:animate-rise">
      {result.warnings.length > 0 && (
        <ul className="mb-3 space-y-1.5">
          {result.warnings.map((warning) => (
            <li key={warning}>
              <Notice>{warning}</Notice>
            </li>
          ))}
        </ul>
      )}

      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-ink text-sm font-medium">
          Structured document
          <span className="text-ink-subtle ml-2 font-normal tabular-nums">
            ({result.record_count} record{result.record_count === 1 ? "" : "s"})
          </span>
        </h3>
      </div>

      <StructureTable records={result.records} columns={columns} />

      {/* What the server decided to extract — informational, not an input. */}
      <details className="border-line rounded-container mt-4 border">
        <summary className="text-ink cursor-pointer px-3 py-2 text-sm font-medium focus-visible:outline-accent focus-visible:outline-2 focus-visible:-outline-offset-2">
          Generated structure
        </summary>
        <ul className="border-line space-y-1 border-t px-3 py-2">
          {columns.map((name) => (
            <li key={name} className="text-ink-muted text-xs">
              <span className="text-ink font-medium">{humanizeFieldName(name)}</span> —{" "}
              {result.schema_used[name]}{" "}
              <code className="text-ink-subtle font-mono">{name}</code>
            </li>
          ))}
        </ul>
      </details>

      <div className="border-line mt-4 border-t pt-4">
        <h3 className="text-ink mb-1 text-sm font-medium">Export</h3>
        <p className="text-ink-muted mb-2 text-xs">
          Download the table above as a new file, in the format you need.
        </p>

        <div className="flex flex-wrap items-center gap-2">
          {STRUCTURE_EXPORT_FORMATS.map((format) => (
            <Button
              key={format.value}
              size="sm"
              disabled={isExporting || aiUnavailable}
              onClick={() => onExport(format.value)}
            >
              {isExporting && exportingFormat === format.value
                ? "Downloading…"
                : `Download ${format.label}`}
            </Button>
          ))}
        </div>

        {exportState.kind === "error" && (
          <div className="mt-3">
            <ErrorBlock message={exportState.message} requestId={exportState.requestId} />
          </div>
        )}
      </div>

      {/* The model identifier was a build detail. "Saved result" stays: it is
          the answer to "why was that instant", and it is what Regenerate acts
          on. */}
      {result.from_cache && (
        <p className="text-ink-subtle mt-4 text-xs">Saved result</p>
      )}
    </div>
  );
}
