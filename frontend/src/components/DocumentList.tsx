/**
 * The document library, and the place documents are watched through processing.
 *
 * While any document is still `pending` or `processing` the list re-fetches on a
 * timer, so a document moves from "Waiting" to "Ready" on screen without the
 * user doing anything. Polling stops as soon as everything has reached a
 * terminal state — there is nothing left to watch, so continuing would be pure
 * waste.
 *
 * It occupies the sidebar, which is narrow and always visible, so each row is
 * built to be scanned rather than read: name, then one line of metadata, then
 * the state. Anything longer than that belongs in the workspace header.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { deleteDocument, listDocuments } from "@/api/documents";
import { DocumentStatusBadge } from "@/components/DocumentStatusBadge";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { Skeleton } from "@/components/ui/Skeleton";
import { failureHint, statusCopy, statusMessage } from "@/lib/documentStatus";
import { formatDuration, secondsSince, useTickingNow } from "@/lib/elapsed";
import { formatBytes, formatDate } from "@/lib/format";
import type { DocumentSummary } from "@/types/api";

/** How often to re-check while work is in flight. */
const POLL_INTERVAL_MS = 1500;

type ListState =
  | { kind: "loading" }
  | { kind: "loaded"; documents: DocumentSummary[]; total: number }
  | { kind: "error"; message: string; requestId: string | null };

interface DocumentListProps {
  /** Bumped by the parent after an upload, to force an immediate refresh. */
  refreshToken: number;
  selectedId: string | null;
  onSelect: (document: DocumentSummary | null) => void;
}

export function DocumentList({
  refreshToken,
  selectedId,
  onSelect,
}: DocumentListProps): React.JSX.Element {
  const [state, setState] = useState<ListState>({ kind: "loading" });
  // Kept in a ref so the polling effect does not restart on every state change.
  const isPollingRef = useRef(false);

  const load = useCallback(async (options?: { quiet?: boolean }): Promise<void> => {
    // A poll must not blank the list out; only an explicit load shows the spinner.
    if (!options?.quiet) setState({ kind: "loading" });

    try {
      const response = await listDocuments({ limit: 50 });
      setState({
        kind: "loaded",
        documents: response.items,
        total: response.total,
      });
    } catch (error) {
      setState({
        kind: "error",
        message:
          error instanceof ApiError ? error.message : "Could not load your documents.",
        requestId: error instanceof ApiError ? error.requestId : null,
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshToken]);

  const documents = state.kind === "loaded" ? state.documents : [];
  const inProgressCount = documents.filter(
    (document) => statusCopy(document.status).inProgress,
  ).length;
  const anyInProgress = inProgressCount > 0;

  useEffect(() => {
    if (!anyInProgress) {
      isPollingRef.current = false;
      return;
    }

    isPollingRef.current = true;
    const timer = setInterval(() => {
      void load({ quiet: true });
    }, POLL_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [anyInProgress, load]);

  // One clock for the whole list; each row subtracts from it. Runs only while
  // something is actually being processed.
  const now = useTickingNow(anyInProgress);

  const remove = useCallback(
    async (document: DocumentSummary): Promise<void> => {
      try {
        await deleteDocument(document.id);
        if (selectedId === document.id) onSelect(null);
        await load({ quiet: true });
      } catch {
        // Refresh regardless: the delete may have succeeded before the error, and the
        // server's view is the one that matters.
        await load({ quiet: true });
      }
    },
    [load, onSelect, selectedId],
  );

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <div className="border-line bg-surface sticky top-0 z-10 flex items-center gap-2 border-b px-3 py-2">
        <h2 className="text-ink-subtle text-xs font-medium tracking-wide uppercase">
          Documents
        </h2>
        {state.kind === "loaded" && (
          <span className="text-ink-subtle text-xs tabular-nums">{state.total}</span>
        )}

        <span className="ml-auto flex items-center gap-2">
          {anyInProgress && (
            <span aria-live="polite" className="text-ink-subtle text-xs tabular-nums">
              {inProgressCount} in progress
            </span>
          )}
          <button
            type="button"
            onClick={() => void load()}
            title="Re-check the library now"
            className="text-ink-subtle hover:text-ink focus-visible:outline-accent text-xs transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1"
          >
            Refresh
          </button>
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {state.kind === "loading" && (
          <div className="p-3">
            <Skeleton lines={4} label="Loading your documents" />
          </div>
        )}

        {state.kind === "error" && (
          <div className="p-3">
            <ErrorBlock message={state.message} requestId={state.requestId} />
          </div>
        )}

        {state.kind === "loaded" && documents.length === 0 && (
          <div className="p-3">
            <EmptyState>
              No documents yet. Add a PDF, Word, Excel, CSV or text file above and it will
              appear here while it is read.
            </EmptyState>
          </div>
        )}

        {state.kind === "loaded" && documents.length > 0 && (
          <ul className="divide-line divide-y">
            {documents.map((document) => (
              <DocumentRow
                key={document.id}
                document={document}
                now={now}
                isSelected={selectedId === document.id}
                onSelect={onSelect}
                onDelete={remove}
              />
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

interface DocumentRowProps {
  document: DocumentSummary;
  /** Shared clock, so a row's elapsed counter costs no timer of its own. */
  now: number;
  isSelected: boolean;
  onSelect: (document: DocumentSummary | null) => void;
  onDelete: (document: DocumentSummary) => void;
}

/**
 * One document.
 *
 * The whole information block is the selection control rather than a separate
 * "Analyze" button — in a narrow column the row is the target a person aims at
 * anyway. Only a `ready` document can be opened, so the block is disabled until
 * then; its text stays fully readable either way.
 */
function DocumentRow({
  document,
  now,
  isSelected,
  onSelect,
  onDelete,
}: DocumentRowProps): React.JSX.Element {
  const copy = statusCopy(document.status);
  const hint = failureHint(document.failure_code);
  const isReady = document.status === "ready";
  const isFailed = document.status === "failed";

  // How long it has been since the upload was accepted. Shown only while work is
  // still in flight, where the question "is this stuck?" is a real one.
  const elapsed = copy.inProgress ? secondsSince(document.created_at, now) : null;

  return (
    <li
      className={`flex items-start gap-1 pr-2 ${
        isSelected ? "bg-accent-soft border-accent border-l-2 pl-1.5" : "pl-2"
      }`}
    >
      <button
        type="button"
        disabled={!isReady}
        aria-current={isSelected ? "true" : undefined}
        onClick={() => onSelect(isSelected ? null : document)}
        title={isReady ? undefined : copy.description}
        // Without this the accessible name is every line in the row read out
        // as one string — filename, size, status, failure reason and date.
        aria-label={`${document.original_filename} — ${copy.label}`}
        className="focus-visible:outline-accent min-w-0 flex-1 py-2 text-left transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2 enabled:cursor-pointer"
      >
        <span
          className="text-ink block truncate text-sm font-medium"
          title={document.original_filename}
        >
          {document.original_filename}
        </span>

        <span className="text-ink-subtle mt-0.5 block truncate text-xs tabular-nums">
          {document.file_type.toUpperCase()} · {formatBytes(document.size_bytes)}
          {document.page_count !== null &&
            ` · ${document.page_count} ${document.page_count === 1 ? "page" : "pages"}`}
        </span>

        <span className="mt-1 flex flex-wrap items-center gap-1.5">
          <DocumentStatusBadge status={document.status} />
          {elapsed !== null && (
            <span className="text-ink-subtle text-xs tabular-nums">
              {formatDuration(elapsed)}
            </span>
          )}
        </span>

        {/* A failed document has to say why here: it can never be opened, so
            this row is the only place the reason will ever appear. */}
        {isFailed && (
          <span className="text-critical mt-1 block text-xs">
            {statusMessage(document.status, document.error_message)}
            {hint !== null && <span className="text-ink-subtle"> {hint}</span>}
          </span>
        )}

        <span className="text-ink-subtle mt-1 block truncate text-xs">
          Added {formatDate(document.created_at)}
        </span>
      </button>

      <button
        type="button"
        onClick={() => onDelete(document)}
        title={`Delete ${document.original_filename}`}
        className="text-ink-subtle hover:text-critical focus-visible:outline-accent mt-2 shrink-0 px-1 text-xs transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1"
      >
        Delete
      </button>
    </li>
  );
}
