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
  /** Why processing failed. Present only when `status` is `"failed"`. */
  error_message: string | null;
  created_at: string;
  updated_at: string;
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
