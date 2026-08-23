/**
 * Lists uploaded documents and follows them through processing.
 *
 * While any document is still `pending` or `processing` the list re-fetches on a timer,
 * so a document moves from "Waiting" to "Ready" on screen without the user doing
 * anything. Polling stops as soon as everything has reached a terminal state — there is
 * nothing to watch, so continuing would be pure waste.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { deleteDocument, listDocuments } from "@/api/documents";
import { DocumentStatusBadge } from "@/components/DocumentStatusBadge";
import { failureHint, statusCopy, statusMessage } from "@/lib/documentStatus";
import type { DocumentSummary } from "@/types/api";

/** How often to re-check while work is in flight. */
const POLL_INTERVAL_MS = 1500;

type ListState =
  | { kind: "loading" }
  | { kind: "loaded"; documents: DocumentSummary[]; total: number }
  | { kind: "error"; message: string; requestId: string | null };

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(isoTimestamp: string): string {
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) return "unknown date";
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

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
          error instanceof ApiError
            ? error.message
            : "Could not load your documents.",
        requestId: error instanceof ApiError ? error.requestId : null,
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshToken]);

  const documents = state.kind === "loaded" ? state.documents : [];
  const anyInProgress = documents.some(
    (document) => statusCopy(document.status).inProgress,
  );

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
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          Documents
          {state.kind === "loaded" && (
            <span className="ml-2 text-sm font-normal text-slate-500 dark:text-slate-400">
              ({state.total})
            </span>
          )}
        </h2>
        <div className="flex items-center gap-3">
          {anyInProgress && (
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Updating…
            </span>
          )}
          <button
            type="button"
            onClick={() => void load()}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {state.kind === "loading" && (
        <p className="text-sm text-slate-500 dark:text-slate-400">Loading…</p>
      )}

      {state.kind === "error" && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950">
          <p className="text-sm text-red-700 dark:text-red-300">{state.message}</p>
          {state.requestId !== null && (
            <p className="mt-2 text-xs text-red-600 dark:text-red-400">
              Reference: <code className="font-mono">{state.requestId}</code>
            </p>
          )}
        </div>
      )}

      {state.kind === "loaded" && documents.length === 0 && (
        <div className="rounded-md border border-dashed border-slate-300 p-8 text-center dark:border-slate-600">
          <p className="text-sm text-slate-600 dark:text-slate-400">
            No documents yet.
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
            Upload a PDF, Word, Excel, CSV or text file to get started.
          </p>
        </div>
      )}

      {state.kind === "loaded" && documents.length > 0 && (
        <ul className="divide-y divide-slate-200 dark:divide-slate-700">
          {documents.map((document) => {
            const hint = failureHint(document.failure_code);
            const isSelected = selectedId === document.id;
            const isReady = document.status === "ready";

            return (
              <li key={document.id} className="py-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium text-slate-900 dark:text-slate-100">
                      {document.original_filename}
                    </p>
                    <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      {document.file_type.toUpperCase()} ·{" "}
                      {formatBytes(document.size_bytes)}
                      {document.page_count !== null &&
                        ` · ${document.page_count} ${
                          document.page_count === 1 ? "page" : "pages"
                        }`}{" "}
                      · added {formatDate(document.created_at)}
                    </p>
                    <p
                      className={`mt-1 text-xs ${
                        document.status === "failed"
                          ? "text-red-600 dark:text-red-400"
                          : "text-slate-600 dark:text-slate-400"
                      }`}
                    >
                      {statusMessage(document.status, document.error_message)}
                    </p>
                    {hint !== null && (
                      <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
                        {hint}
                      </p>
                    )}
                  </div>

                  <div className="flex shrink-0 flex-col items-end gap-2">
                    <DocumentStatusBadge status={document.status} />
                    <div className="flex gap-2">
                      {isReady && (
                        <button
                          type="button"
                          onClick={() => onSelect(isSelected ? null : document)}
                          className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
                        >
                          {isSelected ? "Hide" : "View content"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void remove(document)}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-500 transition hover:bg-red-50 hover:text-red-700 dark:border-slate-600 dark:text-slate-400 dark:hover:bg-red-950 dark:hover:text-red-300"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
