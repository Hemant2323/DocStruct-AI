/**
 * Lists the documents the backend knows about.
 *
 * The list is empty until Milestone 2 adds uploading, but it exercises a real API call
 * end-to-end (typed request, typed response, real error handling) rather than mocking
 * the data, and it renders every processing state correctly so that Milestone 2 has
 * somewhere to show its results.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { listDocuments } from "@/api/documents";
import { DocumentStatusBadge } from "@/components/DocumentStatusBadge";
import { statusMessage } from "@/lib/documentStatus";
import type { DocumentSummary } from "@/types/api";

type ListState =
  | { kind: "loading" }
  | { kind: "loaded"; documents: DocumentSummary[]; total: number }
  | { kind: "error"; message: string; requestId: string | null };

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Format an ISO timestamp for display, tolerating an unparseable value. */
function formatDate(isoTimestamp: string): string {
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return "unknown date";
  }
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function DocumentList(): React.JSX.Element {
  const [state, setState] = useState<ListState>({ kind: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ kind: "loading" });
    try {
      const response = await listDocuments({ limit: 20 });
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
  }, [load]);

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
        <button
          type="button"
          onClick={() => void load()}
          disabled={state.kind === "loading"}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          Refresh
        </button>
      </div>

      {state.kind === "loading" && (
        <p className="text-sm text-slate-500 dark:text-slate-400">Loading…</p>
      )}

      {state.kind === "error" && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950">
          <p className="text-sm text-red-700 dark:text-red-300">
            {state.message}
          </p>
          {state.requestId !== null && (
            <p className="mt-2 text-xs text-red-600 dark:text-red-400">
              Reference:{" "}
              <code className="font-mono">{state.requestId}</code>
            </p>
          )}
        </div>
      )}

      {state.kind === "loaded" && state.documents.length === 0 && (
        <div className="rounded-md border border-dashed border-slate-300 p-8 text-center dark:border-slate-600">
          <p className="text-sm text-slate-600 dark:text-slate-400">
            No documents yet.
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
            Uploading arrives in Milestone 2, together with the parsing pipeline.
          </p>
        </div>
      )}

      {state.kind === "loaded" && state.documents.length > 0 && (
        <ul className="divide-y divide-slate-200 dark:divide-slate-700">
          {state.documents.map((document) => (
            <li
              key={document.id}
              className="flex items-start justify-between gap-4 py-3"
            >
              <div className="min-w-0">
                <p className="truncate font-medium text-slate-900 dark:text-slate-100">
                  {document.original_filename}
                </p>
                <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                  {document.file_type.toUpperCase()} ·{" "}
                  {formatBytes(document.size_bytes)} · added{" "}
                  {formatDate(document.created_at)}
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
              </div>
              <DocumentStatusBadge status={document.status} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
