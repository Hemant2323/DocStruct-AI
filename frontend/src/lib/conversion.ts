/**
 * Display vocabulary for a converted document.
 *
 * The server's enum values are the contract and are never rewritten; these are
 * only how they are shown. Keeping the mapping here means "vocabulary" reads as
 * "Standardised wording" in the audit, the section header and the tooltip
 * alike, rather than in whichever of them was written last.
 */

import type {
  ConversionExportFormat,
  CorrectionConfidence,
  CorrectionKind,
} from "@/types/api";

/** What each correction kind actually does, in a reviewer's words. */
export const CORRECTION_KIND_LABELS: Record<CorrectionKind, string> = {
  case_normalization: "Capitalisation",
  vocabulary: "Standardised wording",
  date_format: "Date format",
  whitespace: "Spacing",
  spelling: "Spelling",
};

/**
 * A kind that arrived unrecognised is shown as itself rather than dropped.
 *
 * `rejected` entries carry `kind` as a plain string — a proposal can be refused
 * precisely *because* it named a kind the server does not implement, and hiding
 * that would hide the reason for the refusal.
 */
export function correctionKindLabel(kind: string): string {
  return CORRECTION_KIND_LABELS[kind as CorrectionKind] ?? kind;
}

export const CONFIDENCE_LABELS: Record<CorrectionConfidence, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
};

export function confidenceLabel(confidence: string): string {
  return CONFIDENCE_LABELS[confidence as CorrectionConfidence] ?? confidence;
}

interface ExportFormat {
  value: ConversionExportFormat;
  label: string;
  hint: string;
}

/**
 * The two documents a person actually takes away, deliberately equal.
 *
 * Neither is the "right" one: a converted document is data to sort and filter, and it
 * is also a report to read and edit — which of those someone wants depends on what
 * they uploaded and what they intend next, and the UI has no way to know. Ranking one
 * above the other would only be a guess wearing the clothes of a recommendation, so
 * they are rendered from one list, with one variant and one width.
 *
 * Order here is presentation order. It is not precedence.
 */
export const DOWNLOAD_FORMATS: ExportFormat[] = [
  {
    value: "xlsx",
    label: "Excel",
    hint: "One worksheet per section, plus the changes and corrections.",
  },
  {
    value: "docx",
    label: "Word",
    hint: "The rendered report, as an editable Word document.",
  },
];

/**
 * Markdown, kept as a quiet third.
 *
 * It stays because the Markdown source is already shown further down the panel, and a
 * reader looking at that text has an obvious next question — how do I keep it. It sits
 * apart from the two above rather than beside them: plain text is what you reach for to
 * paste somewhere, not what you hand to someone.
 *
 * JSON is not offered. It is a debugging view of the whole result, and nobody
 * downloading a converted document wants one; the export endpoint still serves it.
 */
export const MARKDOWN_EXPORT_FORMAT: ExportFormat = {
  value: "md",
  label: "Markdown",
  hint: "The same report as plain text.",
};

/**
 * The extraction's traceability tally, which the Convert page does not show.
 *
 * The backend reports "12 of 33 fields across 9 records could not be traced to a
 * confirmed location" — an accurate and genuinely important fact, delivered in the
 * wrong register and the wrong place. As a banner over the whole document it reads
 * as a defect report about a conversion that is fine, while telling the reader
 * nothing about *which* twelve values it means or what to do next. That question is
 * answered where it can actually be acted on: every unverified value already says
 * "Source not verified" against itself, in the record's Sources list.
 *
 * So this filter changes what is announced, not what is true. Nothing is softened —
 * no value gains a location it did not have, and the same counts stay in the
 * response, the audit and the JSON export for anyone reviewing the run.
 *
 * Matched on the phrase rather than the whole sentence so the numbers may vary, and
 * narrowly enough that it cannot swallow a different warning: "could not be traced"
 * belongs to this one family of provenance tallies and to nothing else the server
 * sends. Every other warning — a truncated document, a section that matched no
 * records, trimmed context — still shows.
 */
const TRACEABILITY_TALLY = /could not be traced/i;

export function visibleConversionWarnings(warnings: readonly string[]): string[] {
  return warnings.filter((warning) => !TRACEABILITY_TALLY.test(warning));
}

/**
 * Shown as guidance, never prefilled — a prefilled box invites sending it unread.
 *
 * Deliberately not tied to a subject area. The example a user reads first teaches them
 * what the feature is for, and one drawn from finance or HR would suggest the tool only
 * handles that kind of document. This one demonstrates the three things any instruction
 * can ask for — grouping, standardising, separating — on a document nobody will mistake
 * for a template.
 */
export const CONVERSION_EXAMPLE =
  "Group the entries by category, standardize how the status values are written, " +
  "list anything still outstanding in its own section, and do not invent missing " +
  "information.";

/**
 * Said once, above the box: this works on whatever you uploaded.
 *
 * The kinds are listed rather than described because a list is the only thing that
 * actually dislodges the assumption that a tool like this is for one sort of file. They
 * are examples, not options — there is nothing to pick, and the system is never told
 * which one applies.
 */
export const CONVERSION_SCOPE_NOTE =
  "Works with whatever you uploaded — a report, a register, a spreadsheet of records, " +
  "notes, specifications. The structure is worked out from the document itself.";
