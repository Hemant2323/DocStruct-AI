/**
 * Structure: turn a document into a table, one row per matching entity.
 *
 * Describing the table wanted in plain language is the primary path — the
 * server works out the fields and reports them back as `schema_used`. Naming
 * the fields by hand is kept as an advanced alternative rather than removed,
 * since it is the only way to pin an exact schema, but it is not what a
 * first-time user is confronted with.
 *
 * Owns its own state for the same reason as `AskPanel`: its request shape and
 * its result — a table of many records, not a flat field list — do not fit the
 * summary/explain/extract flow.
 */

import { useCallback, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { exportStructure, structureDocument } from "@/api/analysis";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBlock } from "@/components/ui/ErrorBlock";
import { FIELD_CLASS } from "@/components/ui/controlStyles";
import { LoadingNote, Skeleton } from "@/components/ui/Skeleton";
import { StructureResultView } from "@/components/workflows/StructureResultView";
import type { RunState } from "@/components/workflows/useInsights";
import { saveBlobAsFile } from "@/lib/download";
import {
  normalizeStructureFieldName,
  STRUCTURE_EXAMPLE,
  STRUCTURE_FIELD_TYPES,
  STRUCTURE_KEY_PATTERN,
} from "@/lib/structureFields";
import type {
  StructureExportFormat,
  StructureFieldType,
  StructureInput,
  StructureResponse,
  StructureSchema,
} from "@/types/api";

interface SchemaFieldRow {
  /** Stable per-row identity for React reconciliation — not part of the submitted schema. */
  id: number;
  name: string;
  type: StructureFieldType;
}

interface StructurePanelProps {
  documentId: string;
  aiUnavailable: boolean;
}

export function StructurePanel({
  documentId,
  aiUnavailable,
}: StructurePanelProps): React.JSX.Element {
  const nextRowId = useRef(1);
  const makeRow = (): SchemaFieldRow => ({
    id: nextRowId.current++,
    name: "",
    type: "string",
  });

  const [instruction, setInstruction] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [fields, setFields] = useState<SchemaFieldRow[]>(() => [makeRow()]);
  const [state, setState] = useState<RunState>({ kind: "idle" });
  const [result, setResult] = useState<StructureResponse | null>(null);
  const [exportState, setExportState] = useState<RunState>({ kind: "idle" });
  const [exportingFormat, setExportingFormat] = useState<StructureExportFormat | null>(
    null,
  );

  const isLoading = state.kind === "loading";
  const isExporting = exportState.kind === "loading";
  const canSubmitInstruction = instruction.trim() !== "" && !isLoading && !aiUnavailable;
  const hasValidSchema =
    fields.length > 0 && fields.every((field) => field.name.trim() !== "");
  const canSubmitSchema = hasValidSchema && !isLoading && !aiUnavailable;

  /**
   * The one place a structure request is made, whichever way it was described.
   *
   * The previous result stays on screen until a new one succeeds — a failed
   * regenerate should not also destroy the table the user already has.
   */
  const generate = useCallback(
    async (input: StructureInput): Promise<void> => {
      setState({ kind: "loading" });
      try {
        const response = await structureDocument(documentId, input);
        setResult(response);
        // A new result invalidates anything the old one's exports reported.
        setExportState({ kind: "idle" });
        setExportingFormat(null);
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
    [documentId],
  );

  const submitInstruction = useCallback(
    async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
      event.preventDefault();
      const trimmed = instruction.trim();
      if (trimmed === "") return;
      await generate({ instruction: trimmed });
    },
    [generate, instruction],
  );

  const submitManualSchema = useCallback(
    async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
      event.preventDefault();
      if (!hasValidSchema) return;

      const normalized = fields.map((field) => ({
        displayName: field.name.trim(),
        key: normalizeStructureFieldName(field.name),
        type: field.type,
      }));

      const unnormalizable = normalized.find(
        (field) => !STRUCTURE_KEY_PATTERN.test(field.key),
      );
      if (unnormalizable !== undefined) {
        setState({
          kind: "error",
          message: `"${unnormalizable.displayName}" can't be turned into a valid field name — use at least one letter or number.`,
          requestId: null,
        });
        return;
      }

      // Two different labels can normalize to the same key, which would
      // silently drop one of them from the schema object.
      const namesByKey = new Map<string, string[]>();
      for (const field of normalized) {
        namesByKey.set(field.key, [
          ...(namesByKey.get(field.key) ?? []),
          field.displayName,
        ]);
      }
      const collisions = [...namesByKey.entries()].filter(([, names]) => names.length > 1);
      if (collisions.length > 0) {
        const detail = collisions
          .map(([key, names]) => `${names.join(" and ")} both normalize to "${key}"`)
          .join("; ");
        setState({
          kind: "error",
          message: `Some field names collide once normalized — ${detail}. Rename one of each so they're distinct.`,
          requestId: null,
        });
        return;
      }

      const schema: StructureSchema = {};
      for (const field of normalized) {
        schema[field.key] = field.type;
      }
      await generate({ schema });
    },
    [fields, generate, hasValidSchema],
  );

  const handleExport = useCallback(
    async (format: StructureExportFormat): Promise<void> => {
      if (result === null) return;

      setExportingFormat(format);
      setExportState({ kind: "loading" });
      try {
        // `schema_used` verbatim from the response — the server looks the cached
        // result up by exactly this, so a schema rebuilt from the displayed table
        // (which cannot recover a field's type, or a field no record happened to
        // state) would 404.
        const file = await exportStructure(documentId, result.schema_used, format);
        const filename = file.filename.includes(".")
          ? file.filename
          : `${file.filename}.${format}`;
        saveBlobAsFile(file.blob, filename);
        setExportState({ kind: "idle" });
      } catch (error) {
        setExportState({
          kind: "error",
          message:
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try again.",
          requestId: error instanceof ApiError ? error.requestId : null,
        });
      } finally {
        setExportingFormat(null);
      }
    },
    [documentId, result],
  );

  const updateFieldName = (id: number, name: string): void => {
    setFields((rows) => rows.map((row) => (row.id === id ? { ...row, name } : row)));
  };

  const updateFieldType = (id: number, type: StructureFieldType): void => {
    setFields((rows) => rows.map((row) => (row.id === id ? { ...row, type } : row)));
  };

  const addField = (): void => {
    setFields((rows) => [...rows, makeRow()]);
  };

  const removeField = (id: number): void => {
    setFields((rows) => (rows.length > 1 ? rows.filter((row) => row.id !== id) : rows));
  };

  return (
    <div className="space-y-4">
      {/* Primary workflow: describe the table you want. */}
      <form onSubmit={(event) => void submitInstruction(event)} className="space-y-2">
        <label
          htmlFor="structure-instruction-input"
          className="text-ink block text-sm font-medium"
        >
          Describe the structured document you want
        </label>
        <textarea
          id="structure-instruction-input"
          value={instruction}
          onChange={(event) => setInstruction(event.target.value)}
          disabled={isLoading || aiUnavailable}
          rows={3}
          placeholder={STRUCTURE_EXAMPLE}
          className={`w-full ${FIELD_CLASS}`}
        />
        <p className="text-ink-muted text-xs">
          For example: “{STRUCTURE_EXAMPLE}” — the fields are worked out from your
          description, so you do not need to list them exactly.
        </p>
        <Button type="submit" variant="primary" disabled={!canSubmitInstruction}>
          {isLoading ? "Generating…" : "Generate structure"}
        </Button>
      </form>

      {/* Advanced alternative: name the fields by hand. */}
      <div className="border-line rounded-container border">
        <button
          type="button"
          onClick={() => setAdvancedOpen((open) => !open)}
          aria-expanded={advancedOpen}
          className="text-ink hover:bg-sunken focus-visible:outline-accent flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2"
        >
          <span>Advanced: define fields manually</span>
          <span className="text-ink-subtle text-xs">{advancedOpen ? "Hide" : "Show"}</span>
        </button>

        {advancedOpen && (
          <form
            onSubmit={(event) => void submitManualSchema(event)}
            className="border-line space-y-3 border-t p-3"
          >
            <p className="text-ink-muted text-xs">
              Name each field yourself instead of describing the document. Use this when
              you need an exact schema — for example to match an earlier export.
            </p>

            <div className="space-y-2">
              {fields.map((field) => (
                <div key={field.id} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={field.name}
                    onChange={(event) => updateFieldName(field.id, event.target.value)}
                    disabled={isLoading || aiUnavailable}
                    placeholder="e.g. account_holder"
                    aria-label="Field name"
                    className={`min-w-0 flex-1 ${FIELD_CLASS}`}
                  />
                  <select
                    value={field.type}
                    onChange={(event) =>
                      updateFieldType(field.id, event.target.value as StructureFieldType)
                    }
                    disabled={isLoading || aiUnavailable}
                    aria-label="Field type"
                    className={FIELD_CLASS}
                  >
                    {STRUCTURE_FIELD_TYPES.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {fields.length > 1 && (
                    <Button
                      variant="subtle"
                      size="sm"
                      onClick={() => removeField(field.id)}
                      disabled={isLoading || aiUnavailable}
                      title="Remove this field"
                      className="shrink-0"
                    >
                      Remove
                    </Button>
                  )}
                </div>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={addField}
                disabled={isLoading || aiUnavailable}
                size="sm"
              >
                Add field
              </Button>
              <Button type="submit" size="sm" disabled={!canSubmitSchema}>
                {isLoading ? "Generating…" : "Generate from these fields"}
              </Button>
              {!hasValidSchema && (
                <span className="text-ink-subtle text-xs">
                  Give every field a name to continue.
                </span>
              )}
            </div>
          </form>
        )}
      </div>

      {isLoading && (
        <div>
          <Skeleton lines={5} label="Reading the document" />
          <LoadingNote>
            Reading the document — this can take up to a minute for a long document.
          </LoadingNote>
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBlock message={state.message} requestId={state.requestId} />
      )}

      {!isLoading && state.kind !== "error" && result === null && (
        <EmptyState>
          Describe the table you want above — for a payment document, one row per
          transaction. Every matching record is extracted, each value cited back to the
          place it came from.
        </EmptyState>
      )}

      {!isLoading && result !== null && (
        <StructureResultView
          result={result}
          aiUnavailable={aiUnavailable}
          isExporting={isExporting}
          exportingFormat={exportingFormat}
          exportState={exportState}
          onExport={handleExport}
        />
      )}
    </div>
  );
}
