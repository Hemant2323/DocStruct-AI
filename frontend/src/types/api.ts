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
