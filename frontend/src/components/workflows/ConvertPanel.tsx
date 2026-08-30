/**
 * Convert: build a new, organized document out of this one.
 *
 * Distinct from Structure, and the difference is worth stating because the two
 * take a similar-looking instruction. Structure answers "what values does this
 * document contain?" and returns one flat table. Convert answers "what document
 * do you want built out of those values?" and returns a section tree, the plan
 * behind it, and an audit of every change proposed along the way.
 *
 * Nothing here is cached: unlike Summary or Extract, the same instruction may
 * legitimately produce a different document, so every run is stored as its own
 * conversion and the history below is how you get back to one. `refresh`
 * therefore controls only whether the *extraction* behind the run is reused.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import {
  convertDocument,
  exportConversion,
  getConversion,
  listConversions,
  MAX_INSTRUCTION_LENGTH,
} from "@/api/conversion";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { LoadingNote, Skeleton } from "@/components/ui/Skeleton";
import { ConversionHistory } from "@/components/workflows/ConversionHistory";
import { ConversionResultView } from "@/components/workflows/ConversionResultView";
import type { RunState } from "@/components/workflows/useInsights";
import { CONVERSION_EXAMPLE, CONVERSION_SCOPE_NOTE } from "@/lib/conversion";
import { saveBlobAsFile } from "@/lib/download";
import type {
  ConversionExportFormat,
  ConversionResponse,
  ConversionSummary,
} from "@/types/api";

interface ConvertPanelProps {
  documentId: string;
  aiUnavailable: boolean;
}

/**
 * Turn a thrown value into the error state.
 *
 * Module-level because it closes over nothing: defined inside the component it
 * was a new function on every render, silently outside the dependency arrays of
 * the callbacks that used it.
 */
function toErrorState(error: unknown): RunState {
  return {
    kind: "error",
    message:
      error instanceof ApiError
        ? error.message
        : "Something went wrong. Please try again.",
    requestId: error instanceof ApiError ? error.requestId : null,
  };
}

export function ConvertPanel({
  documentId,
  aiUnavailable,
}: ConvertPanelProps): React.JSX.Element {
  const [instruction, setInstruction] = useState("");
  const [refresh, setRefresh] = useState(false);
  const [state, setState] = useState<RunState>({ kind: "idle" });
  const [conversion, setConversion] = useState<ConversionResponse | null>(null);
  const [history, setHistory] = useState<ConversionSummary[]>([]);
  const [exportState, setExportState] = useState<RunState>({ kind: "idle" });
  const [exportingFormat, setExportingFormat] = useState<ConversionExportFormat | null>(
    null,
  );

  const isLoading = state.kind === "loading";
  const isExporting = exportState.kind === "loading";
  const trimmed = instruction.trim();
  const tooLong = trimmed.length > MAX_INSTRUCTION_LENGTH;
  const canSubmit = trimmed !== "" && !tooLong && !isLoading && !aiUnavailable;

  const loadHistory = useCallback(async (): Promise<void> => {
    try {
      const response = await listConversions(documentId);
      setHistory(response.items);
    } catch {
      // Not fatal: converting still works, there is just no list to offer. The
      // failure that matters — the conversion itself — has its own error path.
      setHistory([]);
    }
  }, [documentId]);

  // Free on open: listing never calls the AI.
  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const submit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
      event.preventDefault();
      if (trimmed === "" || tooLong) return;

      setState({ kind: "loading" });
      try {
        const result = await convertDocument(documentId, trimmed, refresh);
        setConversion(result);
        // A new conversion invalidates whatever the previous one's exports said.
        setExportState({ kind: "idle" });
        setExportingFormat(null);
        setState({ kind: "idle" });
        // The run was stored server-side, so the list is now one behind.
        await loadHistory();
      } catch (error) {
        setState(toErrorState(error));
      }
    },
    [documentId, loadHistory, refresh, tooLong, trimmed],
  );

  const open = useCallback(
    async (conversionId: string): Promise<void> => {
      setState({ kind: "loading" });
      try {
        const result = await getConversion(documentId, conversionId);
        setConversion(result);
        setExportState({ kind: "idle" });
        setExportingFormat(null);
        setState({ kind: "idle" });
      } catch (error) {
        setState(toErrorState(error));
      }
    },
    [documentId],
  );

  const handleExport = useCallback(
    async (format: ConversionExportFormat): Promise<void> => {
      if (conversion === null) return;

      setExportingFormat(format);
      setExportState({ kind: "loading" });
      try {
        const file = await exportConversion(documentId, conversion.id, format);
        const filename = file.filename.includes(".")
          ? file.filename
          : `${file.filename}.${format}`;
        saveBlobAsFile(file.blob, filename);
        setExportState({ kind: "idle" });
      } catch (error) {
        setExportState(toErrorState(error));
      } finally {
        setExportingFormat(null);
      }
    },
    [conversion, documentId],
  );

  return (
    <div className="space-y-4">
      {/* A composer, not a field in a stack.
          This box is where the whole workflow starts, and the instruction people
          write in it is a sentence or two rather than a value — so it gets the
          room a sentence needs, and the controls that act on it sit inside the
          same frame instead of floating below as unrelated widgets.

          The focus ring is on the frame: the textarea's own outline is removed,
          so focus is shown once, around the thing that actually has it, rather
          than as a box inside a box. Keyed to the textarea specifically and not
          `focus-within` — the button and the checkbox inside carry their own
          rings, and with `focus-within` the frame lit up behind them too. */}
      <form
        onSubmit={(event) => void submit(event)}
        className="border-line bg-surface rounded-container has-[textarea:focus]:outline-accent overflow-hidden border transition-colors duration-100 has-[textarea:focus]:outline-2 has-[textarea:focus]:outline-offset-1"
      >
        <div className="px-4 pt-4">
          <label
            htmlFor="convert-instruction-input"
            className="text-ink block text-sm font-semibold"
          >
            Describe the document you want built
          </label>
          <p className="text-ink-subtle mt-1 text-xs leading-relaxed">
            {CONVERSION_SCOPE_NOTE}
          </p>

          <textarea
            id="convert-instruction-input"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            disabled={isLoading || aiUnavailable}
            rows={4}
            placeholder={CONVERSION_EXAMPLE}
            aria-describedby="convert-instruction-help"
            className="text-ink placeholder:text-ink-subtle mt-3 w-full resize-y border-0 bg-transparent p-0 text-sm leading-relaxed focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
          />

          <p
            id="convert-instruction-help"
            className="text-ink-muted border-line mt-2 border-t pt-2 text-xs leading-relaxed"
          >
            Say how it should be organised, what to standardise and what to separate out.
            Nothing is invented: every value in the result is your source text, or a
            recorded correction of it.
          </p>
        </div>

        {/* The action bar. Option on the left, the act on the right — the order
            in which the two are decided. */}
        <div className="border-line bg-sunken/60 mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t px-4 py-3">
          <label className="flex max-w-md cursor-pointer items-start gap-2.5">
            <input
              type="checkbox"
              checked={refresh}
              onChange={(event) => setRefresh(event.target.checked)}
              disabled={isLoading || aiUnavailable}
              className="accent-accent mt-0.5"
            />
            <span className="text-xs">
              <span className="text-ink font-medium">Re-read the source document</span>
              <span className="text-ink-muted mt-0.5 block leading-relaxed">
                Extract the source again instead of reusing the last extraction. Slower
                and billed; the conversion itself is produced fresh either way.
              </span>
            </span>
          </label>

          <div className="ml-auto flex items-center gap-3">
            {/* Counted only near the limit, where the number is information
                rather than pressure. The server enforces the same bound. */}
            {trimmed.length > MAX_INSTRUCTION_LENGTH * 0.8 && (
              <span
                className={`text-xs tabular-nums ${tooLong ? "text-critical" : "text-ink-subtle"}`}
              >
                {trimmed.length} / {MAX_INSTRUCTION_LENGTH}
                {tooLong && " — too long"}
              </span>
            )}

            <Button type="submit" variant="primary" disabled={!canSubmit}>
              {isLoading ? "Building…" : "Convert document"}
            </Button>
          </div>
        </div>
      </form>

      {isLoading && (
        <div className="border-line rounded-container border p-4">
          <Skeleton lines={6} label="Building the document" />
          <LoadingNote>
            Reading the source, planning the layout, then checking every proposed
            correction — this usually takes under a minute.
          </LoadingNote>
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBlock message={state.message} requestId={state.requestId} />
      )}

      {!isLoading && state.kind !== "error" && conversion === null && (
        <EmptyState>
          Turn this document into a new, organized one — grouped how you ask, with
          inconsistent values standardised, ready to download as a spreadsheet or a Word
          document. Every
          change is listed, and anything the system could not prove is left exactly as
          your document wrote it.
        </EmptyState>
      )}

      {!isLoading && conversion !== null && (
        // The reveal is on the result, not the panel: it marks the arrival of
        // something new, and re-running the animation on every keystroke in the
        // box above would be noise.
        <ConversionResultView
          key={conversion.id}
          conversion={conversion}
          isExporting={isExporting}
          exportingFormat={exportingFormat}
          exportState={exportState}
          onExport={handleExport}
        />
      )}

      <ConversionHistory
        items={history}
        activeId={conversion?.id ?? null}
        isLoading={isLoading}
        onOpen={(conversionId) => void open(conversionId)}
      />
    </div>
  );
}
