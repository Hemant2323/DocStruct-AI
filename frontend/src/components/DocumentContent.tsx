/**
 * Shows the structured content extracted from one document.
 *
 * Every block displays its `location_label` — "page 4", "lines 40-48",
 * "Q3 Budget!A12:F30". That label is the visible proof that extraction kept
 * track of where each piece came from, which is what makes a citation something
 * the system can resolve rather than something the model asserts.
 *
 * Fifty blocks at a time, so the presentation is a divided list inside one
 * border rather than fifty bordered cards: the blocks are one sequence, and
 * boxing each of them said they were separate things.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { getDocumentContent } from "@/api/documents";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { Skeleton } from "@/components/ui/Skeleton";
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
  | { kind: "error"; message: string; requestId: string | null };

interface DocumentContentProps {
  documentId: string;
}

export function DocumentContent({ documentId }: DocumentContentProps): React.JSX.Element {
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
        // Kept so this failure can be quoted like every other one in the app —
        // it is the only key to the matching server log line.
        requestId: error instanceof ApiError ? error.requestId : null,
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
    <section className="px-6 py-5">
      <div className="mb-3">
        <h2 className="text-ink text-sm font-semibold">Extracted content</h2>
        {state.kind === "loaded" && (
          <p className="text-ink-subtle mt-0.5 text-xs tabular-nums">
            {total} extracted {total === 1 ? "section" : "sections"}
          </p>
        )}
      </div>

      {state.kind === "loading" && <Skeleton lines={5} label="Loading extracted content" />}

      {state.kind === "error" && (
        <ErrorBlock message={state.message} requestId={state.requestId} />
      )}

      {state.kind === "loaded" && state.blocks.length === 0 && (
        <EmptyState>No content has been extracted yet.</EmptyState>
      )}

      {state.kind === "loaded" && state.blocks.length > 0 && (
        <>
          <ul className="divide-line border-line rounded-container divide-y border">
            {state.blocks.map((block) => (
              <li key={block.order} className="px-3 py-2.5">
                <p className="text-ink-subtle mb-1 flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="font-medium tracking-wide uppercase">
                    {BLOCK_LABELS[block.block_type]}
                    {block.level !== null && ` ${block.level}`}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono">{block.location_label}</span>
                  {!block.is_exact && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span
                        title="This location was inferred rather than declared by the file, so it is approximate."
                        className="text-caution"
                      >
                        approximate
                      </span>
                    </>
                  )}
                </p>

                <div
                  // A laid-out block (a table, a batch of sheet rows) keeps its
                  // line breaks and scrolls, so it needs to be focusable to be
                  // scrollable without a pointer. Prose wraps instead, including
                  // inside a long unbroken token — a 200-character URL must not
                  // widen the page.
                  {...(MONOSPACE_TYPES.has(block.block_type)
                    ? { role: "region", "aria-label": block.location_label, tabIndex: 0 }
                    : {})}
                  className={`text-ink focus-visible:outline-accent overflow-x-auto text-sm focus-visible:outline-2 focus-visible:-outline-offset-2 ${
                    MONOSPACE_TYPES.has(block.block_type)
                      ? "font-mono text-xs whitespace-pre"
                      : "wrap-anywhere whitespace-pre-wrap"
                  }`}
                >
                  {block.text}
                </div>
              </li>
            ))}
          </ul>

          {(hasPrevious || hasNext) && (
            <div className="mt-3 flex items-center justify-between gap-4">
              <Button
                size="sm"
                disabled={!hasPrevious}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                Previous
              </Button>
              <span className="text-ink-subtle text-xs tabular-nums">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
              </span>
              <Button size="sm" disabled={!hasNext} onClick={() => setOffset(offset + PAGE_SIZE)}>
                Next
              </Button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
