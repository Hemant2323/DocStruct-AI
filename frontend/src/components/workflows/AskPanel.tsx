/**
 * Ask: a question about the document, answered from its content.
 *
 * Kept apart from the Summary/Explain/Extract wiring on purpose. An answer is
 * never cached or pre-loaded — two questions share nothing to cache against —
 * so this panel owns its own state rather than joining theirs.
 *
 * The previous answer stays on screen while a new question is in flight, and
 * survives a failed one: losing an answer you were reading because the next
 * request failed would be its own small betrayal.
 */

import { useCallback, useState } from "react";

import { askQuestion } from "@/api/analysis";
import { ApiError } from "@/api/client";
import { CitationList } from "@/components/CitationList";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { FIELD_CLASS } from "@/components/ui/controlStyles";
import { LoadingNote, Skeleton } from "@/components/ui/Skeleton";
import type { RunState } from "@/components/workflows/useInsights";
import type { AskResponse } from "@/types/api";

interface AskPanelProps {
  documentId: string;
  aiUnavailable: boolean;
}

export function AskPanel({ documentId, aiUnavailable }: AskPanelProps): React.JSX.Element {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<RunState>({ kind: "idle" });
  const [result, setResult] = useState<AskResponse | null>(null);

  const isLoading = state.kind === "loading";
  const canSubmit = question.trim() !== "" && !isLoading && !aiUnavailable;

  const submit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
      event.preventDefault();
      const trimmed = question.trim();
      if (trimmed === "") return;

      setState({ kind: "loading" });
      try {
        const answer = await askQuestion(documentId, trimmed);
        setResult(answer);
        setState({ kind: "idle" });
      } catch (error) {
        setState({
          kind: "error",
          message:
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try again.",
          requestId: error instanceof ApiError ? error.requestId : null,
        });
      }
    },
    [documentId, question],
  );

  return (
    <div className="space-y-4">
      <form onSubmit={(event) => void submit(event)} className="space-y-2">
        <label htmlFor="ask-question-input" className="text-ink block text-sm font-medium">
          Ask a question about this document
        </label>
        <textarea
          id="ask-question-input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={isLoading || aiUnavailable}
          rows={2}
          placeholder="e.g. What is the payment deadline?"
          className={`w-full ${FIELD_CLASS}`}
        />
        <Button type="submit" variant="primary" disabled={!canSubmit}>
          {isLoading ? "Asking…" : "Ask"}
        </Button>
      </form>

      {isLoading && (
        <div>
          <Skeleton lines={3} label="Thinking" />
          <LoadingNote>
            Thinking — this can take up to a minute for a long document.
          </LoadingNote>
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBlock message={state.message} requestId={state.requestId} />
      )}

      {!isLoading && state.kind !== "error" && result === null && (
        <EmptyState>
          Ask anything about this document — the answer is grounded in its content, with
          sources you can check. Each question is answered fresh rather than saved.
        </EmptyState>
      )}

      {!isLoading && result !== null && (
        // The model identifier and the count of document sections fed to it
        // were both build details: nothing a reader decides changes either, and
        // the sources below already say what the answer rests on.
        <div className="motion-safe:animate-rise">
          <p className="text-ink text-sm leading-relaxed wrap-anywhere whitespace-pre-wrap">
            {result.answer}
          </p>
          <CitationList citations={result.citations} />
        </div>
      )}
    </div>
  );
}
