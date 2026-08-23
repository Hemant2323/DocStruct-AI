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
