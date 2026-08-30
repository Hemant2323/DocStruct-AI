/**
 * Field naming and format options for the Structure workflow.
 *
 * The backend embeds schema keys directly in the prompt it sends the model, so
 * it restricts them to `^[a-z][a-z0-9_]{0,63}$`. People type labels, not keys —
 * so the two are kept apart here: the label stays on screen, the key is what
 * travels, and neither is ever silently substituted for the other.
 */

import type { StructureExportFormat, StructureFieldType } from "@/types/api";

/** Field types offered by the schema builder. Mirrors the backend's supported types. */
export const STRUCTURE_FIELD_TYPES: { value: StructureFieldType; label: string }[] = [
  { value: "string", label: "String" },
  { value: "number", label: "Number" },
  { value: "integer", label: "Integer" },
  { value: "boolean", label: "Boolean" },
  { value: "date", label: "Date" },
];

/** Matches the backend's field-name contract exactly (`dynamic_schema.py`'s `_FIELD_NAME_RE`). */
export const STRUCTURE_KEY_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;

/** File formats offered by the export section. */
export const STRUCTURE_EXPORT_FORMATS: { value: StructureExportFormat; label: string }[] =
  [
    { value: "json", label: "JSON" },
    { value: "csv", label: "CSV" },
    { value: "xlsx", label: "XLSX" },
  ];

/** Shown as guidance, never prefilled — a prefilled box invites sending it unread. */
export const STRUCTURE_EXAMPLE =
  "Create a payment report with one row per transaction. Include account holder, " +
  "bank, amount, deadline and status.";

/**
 * Turn a user-typed field label into a valid backend schema key.
 *
 * "Account Holder" → `account_holder`. The original label stays visible in the
 * UI and is never itself sent.
 */
export function normalizeStructureFieldName(displayName: string): string {
  const withUnderscores = displayName.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
  const startingAtLetter = withUnderscores.replace(/^[^a-z]+/, "");
  return startingAtLetter.slice(0, 64);
}

/**
 * `account_holder` → `Account Holder`, for display only.
 *
 * The schema key itself is never rewritten: it is what the server extracted
 * under and what `/structure/export` is looked up by, so only the label a
 * person reads changes.
 */
export function humanizeFieldName(name: string): string {
  return name
    .split("_")
    .filter((part) => part !== "")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
