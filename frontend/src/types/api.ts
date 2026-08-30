/**
 * Types mirroring the backend's API contract.
 *
 * These are hand-written to match the FastAPI response models. Keeping them in one file
 * means a backend contract change is a single, reviewable edit here rather than a hunt
 * through components.
 */

/** Lifecycle of a document in the ingestion pipeline. Mirrors `ProcessingStatus`. */
export type ProcessingStatus = "pending" | "processing" | "ready" | "failed";

/** File types the platform can ingest. Mirrors `SupportedFileType`. */
export type SupportedFileType = "pdf" | "docx" | "txt" | "csv" | "xlsx";

/** A document's metadata and processing status. */
export interface DocumentSummary {
  id: string;
  original_filename: string;
  file_type: SupportedFileType;
  size_bytes: number;
  status: ProcessingStatus;
  /** Why processing failed, in plain language. Present only when `status` is `"failed"`. */
  error_message: string | null;
  /**
   * Machine-readable failure reason. Branch on this rather than on `error_message`,
   * which may be reworded. Known values include `requires_ocr`, `encrypted_file`,
   * `corrupt_file`, `no_content`, `too_complex` and `interrupted`.
   */
  failure_code: string | null;
  /** Pages for a document, worksheets for a workbook. Null until processing succeeds. */
  page_count: number | null;
  /** When extraction finished, successfully or not. */
  processed_at: string | null;
  created_at: string;
  updated_at: string;
}

/** What kind of content an extracted block holds. */
export type BlockType =
  | "heading"
  | "paragraph"
  | "list_item"
  | "table"
  | "schema"
  | "sheet_row"
  | "caption";

/** One extracted piece of content, with where it came from. */
export interface ContentBlock {
  order: number;
  block_type: BlockType;
  text: string;
  level: number | null;
  /**
   * Where this came from in the original file — "page 4", "lines 40-48",
   * "Q3 Budget!A12:F30". This is what a citation displays.
   */
  location_label: string;
  /** False when the location is approximate, e.g. a PDF heading inferred from font size. */
  is_exact: boolean;
  char_count: number;
}

/** A page of a document's extracted content. */
export interface ContentResponse {
  items: ContentBlock[];
  total: number;
  limit: number;
  offset: number;
}

/** A page of documents. */
export interface DocumentListResponse {
  items: DocumentSummary[];
  total: number;
  limit: number;
  offset: number;
}

/** Liveness probe response. */
export interface HealthResponse {
  status: "ok";
  app: string;
  version: string;
  environment: string;
}

/** Readiness probe response. */
export interface ReadinessResponse {
  status: "ready" | "not_ready";
  checks: Record<string, boolean>;
}

/** The body of every error response from the API. */
export interface ApiErrorBody {
  error: {
    /** Stable machine-readable code. Branch on this, not on `message`. */
    code: string;
    /** Human-readable description, safe to show to the user. */
    message: string;
    /** Structured context, e.g. which fields failed validation. */
    details: Record<string, unknown>;
    /**
     * Correlation ID matching the `X-Request-ID` response header. Show it to the user
     * on an unexpected failure: the server log holds the technical detail under it.
     */
    request_id: string | null;
  };
}

// --- AI analysis (Milestone 3) ---------------------------------------------

/** What kind of AI result this is. */
export type InsightKind = "summary" | "explanation" | "extraction";

/** How the document was presented to the AI. */
export type ContextMode = "full" | "optimized";

/** Where a piece of an AI result came from. */
export interface Citation {
  /** The exact wording the result is based on. */
  quoted_text: string;
  /**
   * Where that wording appears in the original file — "page 4", "lines 40-48",
   * "Q3 Budget!A12:F30". Reads "source not verified" when it could not be confirmed.
   */
  location_label: string;
  /**
   * True when the location was resolved from stored content. False means it could not
   * be confirmed — a location is never guessed or invented.
   */
  is_verified: boolean;
  block_ids: string[];
}

/** One fact extracted from a document. */
export interface ExtractedField {
  name: string;
  value: string;
  /** Null when the supporting quote could not be found in the document. */
  citation: Citation | null;
}

/** An AI result about a document. */
export interface Insight {
  kind: InsightKind;
  /** The prose result. Empty for extractions. */
  content: string;
  /** Extracted facts. Empty for prose results. */
  fields: ExtractedField[];
  citations: Citation[];
  context_mode: ContextMode;
  model: string;
  input_tokens: number;
  output_tokens: number;
  /** Things the user should know — content left out, sources not confirmed. */
  warnings: string[];
  verified_citation_count: number;
  /** True when this was a previously generated result rather than a new one. */
  from_cache: boolean;
  created_at: string | null;
}

export interface InsightListResponse {
  items: Insight[];
}

/** How large a document is for the AI, without calling it. */
export interface ContextPreview {
  chunk_count: number;
  estimated_tokens: number;
  fits_in_full: boolean;
  /** True when this document is large enough that optimizing is worth offering. */
  optimization_available: boolean;
}

/** Whether AI features are usable on this server. */
export interface AIStatus {
  configured: boolean;
  model: string;
}

/** Options accepted by the analysis endpoints. */
export interface AnalysisOptions {
  optimize_context?: boolean;
  refresh?: boolean;
  extraction_type?: string;
}

// --- Question answering (Milestone 4.4) -------------------------------------

/** A question to ask about a document. */
export interface AskRequest {
  question: string;
  /** Maximum number of document sections to retrieve and consider. Server default: 5. */
  top_k?: number;
}

/** An answer to a question about a document, grounded in its content. */
export interface AskResponse {
  question: string;
  /** States plainly when the document does not contain enough information, rather than guessing. */
  answer: string;
  citations: Citation[];
  model: string;
  input_tokens: number;
  output_tokens: number;
  /** How many document sections were used to answer. */
  chunk_count: number;
}

// --- Caller-defined structured extraction (Milestone 4.5 / 4.6) -------------

/** A field type supported by the `/structure` endpoint. */
export type StructureFieldType = "string" | "number" | "integer" | "boolean" | "date";

/**
 * A caller-defined set of fields to extract, as a flat mapping of field name to type.
 * Field names must be lowercase, start with a letter, and contain only letters, numbers
 * and underscores (max 64 characters, max 40 fields) — enforced by the backend.
 */
export type StructureSchema = Record<string, StructureFieldType>;

/** File formats `/structure/export` can produce. */
export type StructureExportFormat = "json" | "csv" | "xlsx";

/**
 * One record from a structure result — a single transaction, person, or other entity
 * the document describes.
 *
 * Deliberately the existing {@link ExtractedField}: the backend returns each record as a
 * list of exactly that shape (`name`, `value`, `citation`), so a field inside a record
 * and a field anywhere else in the API are the same thing. A field the document does not
 * state for this particular record is simply absent from the list rather than present
 * with an empty value.
 */
export type StructureRecord = ExtractedField[];

/**
 * What to extract, given either as an explicit schema or as a plain-language
 * description. Exactly one — the union makes sending both unrepresentable rather than
 * relying on the server to reject it.
 */
export type StructureInput =
  | { instruction: string }
  | { schema: StructureSchema };

/** The body sent to `POST /documents/{id}/structure`. */
export type StructureRequest = StructureInput & { refresh?: boolean };

/** A structured, multi-record result: one record per matching entity in a document. */
export interface StructureResponse {
  kind: InsightKind;
  document_id: string;
  /** Every matching record found. One entry per transaction/entity, not just the first. */
  records: StructureRecord[];
  /**
   * The exact schema this result was produced with — as supplied, or as inferred from
   * an instruction. This is what determines table column order, and what must be sent
   * back to `/structure/export`; never rebuild it from the displayed records.
   */
  schema_used: StructureSchema;
  record_count: number;
  /** Every confirmed source across all records. */
  citations: Citation[];
  context_mode: ContextMode;
  model: string;
  input_tokens: number;
  output_tokens: number;
  warnings: string[];
  verified_citation_count: number;
  from_cache: boolean;
  created_at: string | null;
}

/* ---------------------------------------------------------------------------
   Document conversion

   Conversion is not structure. Structure answers "what values does this
   document contain?" and returns one flat table. Conversion answers "what
   document did the user ask to have built out of those values?" and returns a
   section tree, the plan behind it, and an audit of everything that was changed
   — which is why it has its own shapes rather than widening the structure ones.
--------------------------------------------------------------------------- */

/**
 * Where a value in a converted document came from.
 *
 * There is deliberately no member meaning "text that appeared from nowhere":
 * a field the source does not state is `absent` and renders blank.
 */
export type ValueOrigin = "source" | "corrected" | "structural" | "absent";

/** How sure a proposed correction was. Only `high` is ever applied. */
export type CorrectionConfidence = "high" | "medium" | "low";

/**
 * What sort of change a correction makes. Each kind has its own deterministic
 * proof on the server; a correction that cannot prove itself is refused
 * whatever confidence it claimed.
 */
export type CorrectionKind =
  | "case_normalization"
  | "vocabulary"
  | "date_format"
  | "whitespace"
  | "spelling";

/** How a section presents its records. */
export type ConversionSectionKind = "table" | "detail" | "note";

/** File formats a conversion can be downloaded as. */
export type ConversionExportFormat = "md" | "docx" | "xlsx" | "json";

/** One value in a converted document, with its origin and provenance. */
export interface ConvertedValue {
  text: string;
  origin: ValueOrigin;
  /** The text this replaced. Present only when `origin` is `"corrected"`. */
  original_text: string | null;
  /** The audit entry justifying the change. Present only when corrected. */
  correction_id: string | null;
  /** Where the original appears in the source. Null when unconfirmed; never guessed. */
  source_label: string | null;
  quoted_text: string | null;
  is_verified: boolean;
}

/** One named field of a record. `name` is the key; `label` is display only. */
export interface ConvertedField {
  name: string;
  label: string;
  value: ConvertedValue;
}

/** One entity in the converted document, traced back to the extraction it came from. */
export interface ConvertedRecord {
  source_record_index: number;
  fields: ConvertedField[];
}

/** One section of the converted document. */
export interface ConvertedSection {
  heading: string;
  kind: ConversionSectionKind;
  /** Field names to show, in order. Empty for a note. */
  columns: string[];
  records: ConvertedRecord[];
  /** Prose body, for a `note` section. */
  note: string;
}

/** A change that was applied, or one that was proposed and left unapplied. */
export interface ConversionCorrection {
  id: string;
  kind: CorrectionKind;
  record_index: number;
  field: string;
  original: string;
  corrected: string;
  reason: string;
  confidence: CorrectionConfidence;
  /** True when the document shows the correction; false when it shows the original. */
  applied: boolean;
}

/**
 * A proposed change that was refused.
 *
 * Surfaced rather than discarded: that the system was asked to alter a value and
 * declined is something a reviewer of a financial document should be able to see.
 */
export interface ConversionRejection {
  kind: string;
  record_index: number;
  field: string;
  original: string;
  corrected: string;
  confidence: string;
  rejection_reason: string;
}

/** One section of the layout that produced the document. */
export interface ConversionPlanSection {
  heading: string;
  kind: ConversionSectionKind;
  columns: string[];
  group_by: string | null;
  note: string;
}

/** The layout the document was built from, after invalid parts were stripped. */
export interface ConversionPlan {
  title: string;
  sections: ConversionPlanSection[];
  labels: Record<string, string>;
}

/** One past conversion, without its content — for listing. */
export interface ConversionSummary {
  id: string;
  document_id: string;
  title: string;
  instruction: string;
  source_record_count: number;
  applied_correction_count: number;
  suggestion_count: number;
  rejected_correction_count: number;
  created_at: string;
}

/** Every conversion produced for a document, newest first. */
export interface ConversionListResponse {
  items: ConversionSummary[];
}

/** The body sent to `POST /documents/{id}/convert`. */
export interface ConversionRequest {
  instruction: string;
  /**
   * Re-read the source instead of reusing its cached extraction. Each request
   * produces a new conversion either way; this controls only the extraction.
   */
  refresh?: boolean;
}

/** A completed conversion: the new document, and everything that justifies it. */
export interface ConversionResponse {
  id: string;
  document_id: string;
  title: string;
  instruction: string;
  /** The rendered document, ready to display or download. */
  markdown: string;
  /** The same document in structured form. */
  sections: ConvertedSection[];
  plan: ConversionPlan;
  /** Changes that were applied. */
  corrections: ConversionCorrection[];
  /** Proposed but not applied — the document still shows the original values. */
  suggestions: ConversionCorrection[];
  /** Refused as unsafe, with reasons. */
  rejected: ConversionRejection[];
  schema_used: Record<string, string>;
  vocabularies: Record<string, string[]>;
  source_record_count: number;
  warnings: string[];
  model: string;
  input_tokens: number;
  output_tokens: number;
  /** Whether the extraction behind this was reused. The conversion is always fresh. */
  records_from_cache: boolean;
  created_at: string;
}
