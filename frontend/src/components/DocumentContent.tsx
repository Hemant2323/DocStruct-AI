/**
 * Shows the structured content extracted from one document.
 *
 * Every block displays its `location_label` — "page 4", "lines 40-48",
 * "Q3 Budget!A12:F30". That label is the visible proof that extraction kept track of
 * where each piece came from, which is what makes citations possible in a later
 * milestone rather than something bolted on afterwards.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { getDocumentContent } from "@/api/documents";
import type { BlockType, ContentBlock } from "@/types/api";

const PAGE_SIZE = 50;

/** Short human labels for block types. */
const BLOCK_LABELS: Record<BlockType, string> = {
  heading: "Heading",
  paragraph: "Paragraph",
  list_item: "List item",
  table: "Table",
  schema: "Table structure",
  sheet_row: "Rows",
  caption: "Caption",
};

/** Blocks whose text is laid out and must keep its line breaks. */
const MONOSPACE_TYPES = new Set<BlockType>(["table", "schema", "sheet_row"]);

type ContentState =
  | { kind: "loading" }
  | { kind: "loaded"; blocks: ContentBlock[]; total: number }
  | { kind: "error"; message: string };

interface DocumentContentProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

export function DocumentContent({
  documentId,
  filename,
  onClose,
}: DocumentContentProps): React.JSX.Element {
  const [state, setState] = useState<ContentState>({ kind: "loading" });
  const [offset, setOffset] = useState(0);

  const load = useCallback(async (): Promise<void> => {
    setState({ kind: "loading" });
    try {
      const response = await getDocumentContent(documentId, {
        limit: PAGE_SIZE,
        offset,
      });
      setState({
        kind: "loaded",
        blocks: response.items,
        total: response.total,
      });
    } catch (error) {
      setState({
        kind: "error",
        message:
          error instanceof ApiError
            ? error.message
            : "Could not load this document's content.",
      });
    }
  }, [documentId, offset]);

  useEffect(() => {
    void load();
  }, [load]);

  const total = state.kind === "loaded" ? state.total : 0;
  const hasPrevious = offset > 0;
  const hasNext = offset + PAGE_SIZE < total;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold text-slate-900 dark:text-slate-100">
            {filename}
          </h2>
          {state.kind === "loaded" && (
            <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
              {total} extracted {total === 1 ? "section" : "sections"}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          Close
        </button>
      </div>

      {state.kind === "loading" && (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Loading content…
        </p>
      )}

      {state.kind === "error" && (
        <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {state.message}
        </p>
      )}

      {state.kind === "loaded" && state.blocks.length === 0 && (
        <p className="rounded-md border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600 dark:border-slate-600 dark:text-slate-400">
          No content has been extracted yet.
        </p>
      )}

      {state.kind === "loaded" && state.blocks.length > 0 && (
        <>
          <ul className="space-y-3">
            {state.blocks.map((block) => (
              <li
                key={block.order}
                className="rounded-md border border-slate-200 p-3 dark:border-slate-700"
              >
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <span className="rounded bg-slate-100 px-2 py-0.5 font-medium text-slate-700 dark:bg-slate-700 dark:text-slate-200">
                    {BLOCK_LABELS[block.block_type]}
                    {block.level !== null && ` ${block.level}`}
                  </span>
                  <span className="font-mono text-slate-500 dark:text-slate-400">
                    {block.location_label}
                  </span>
                  {!block.is_exact && (
                    <span
                      title="This location was inferred rather than declared by the file, so it is approximate."
                      className="rounded bg-amber-100 px-2 py-0.5 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                    >
                      approximate
                    </span>
                  )}
                </div>
                <div
                  className={`overflow-x-auto text-sm text-slate-800 dark:text-slate-200 ${
                    MONOSPACE_TYPES.has(block.block_type)
                      ? "whitespace-pre font-mono text-xs"
                      : "whitespace-pre-wrap"
                  }`}
                >
                  {block.text}
                </div>
              </li>
            ))}
          </ul>

          {(hasPrevious || hasNext) && (
            <div className="mt-4 flex items-center justify-between gap-4">
              <button
                type="button"
                disabled={!hasPrevious}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                Previous
              </button>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
              </span>
              <button
                type="button"
                disabled={!hasNext}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
