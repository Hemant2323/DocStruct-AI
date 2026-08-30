/**
 * Identifies the document the workspace is showing.
 *
 * Sticky, because the workflows below it are long: after scrolling through a
 * structured table or a corrections audit, "which document is this?" should
 * still be answerable without scrolling back.
 *
 * Metadata only, plus the one action that belongs to the workspace rather than
 * to the library — closing the document. Deleting stays in the list, where the
 * document is one of several and the consequence is obvious.
 */

import { DocumentStatusBadge } from "@/components/DocumentStatusBadge";
import { Button } from "@/components/ui/Button";
import { failureHint, statusMessage } from "@/lib/documentStatus";
import { formatBytes } from "@/lib/format";
import type { DocumentSummary } from "@/types/api";

interface DocumentHeaderProps {
  document: DocumentSummary;
  onClose: () => void;
}

export function DocumentHeader({
  document,
  onClose,
}: DocumentHeaderProps): React.JSX.Element {
  const hint = failureHint(document.failure_code);
  const isFailed = document.status === "failed";

  return (
    <div className="border-line bg-surface sticky top-0 z-10 border-b px-6 py-3">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <h1 className="text-ink truncate text-sm font-semibold" title={document.original_filename}>
              {document.original_filename}
            </h1>
            <DocumentStatusBadge status={document.status} />
          </div>

          <p className="text-ink-subtle mt-0.5 truncate text-xs tabular-nums">
            {document.file_type.toUpperCase()} · {formatBytes(document.size_bytes)}
            {document.page_count !== null &&
              ` · ${document.page_count} ${document.page_count === 1 ? "page" : "pages"}`}
          </p>
        </div>

        <Button variant="subtle" size="sm" onClick={onClose} className="shrink-0">
          Close
        </Button>
      </div>

      {/* A failed document must say why here: the workspace below it will have
          nothing to show, and an empty panel is not an explanation. */}
      {isFailed && (
        <p className="text-critical mt-2 text-xs">
          {statusMessage(document.status, document.error_message)}
          {hint !== null && <span className="text-ink-subtle"> {hint}</span>}
        </p>
      )}
    </div>
  );
}
