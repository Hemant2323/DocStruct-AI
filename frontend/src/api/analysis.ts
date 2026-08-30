/** Typed wrappers around the AI analysis endpoints. */

import { apiRequest, apiRequestBlob, type DownloadedFile } from "@/api/client";
import type {
  AIStatus,
  AnalysisOptions,
  AskRequest,
  AskResponse,
  ContextPreview,
  Insight,
  InsightListResponse,
  StructureExportFormat,
  StructureInput,
  StructureRequest,
  StructureResponse,
  StructureSchema,
} from "@/types/api";

/**
 * Summarise a document.
 *
 * POST rather than GET because it can cost money and is not safely repeatable. The
 * server caches the result, so calling twice does not bill twice unless `refresh` is set.
 */
export async function generateSummary(
  id: string,
  options: AnalysisOptions = {},
): Promise<Insight> {
  return apiRequest<Insight>(`/documents/${id}/summary`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

/** Explain a document in plain language. */
export async function generateExplanation(
  id: string,
  options: AnalysisOptions = {},
): Promise<Insight> {
  return apiRequest<Insight>(`/documents/${id}/explanation`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

/** Extract structured facts from a document. */
export async function generateExtraction(
  id: string,
  options: AnalysisOptions = {},
): Promise<Insight> {
  return apiRequest<Insight>(`/documents/${id}/extraction`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

/** Fetch AI results already generated for a document. Free — never calls the AI. */
export async function listInsights(id: string): Promise<InsightListResponse> {
  return apiRequest<InsightListResponse>(`/documents/${id}/insights`);
}

/**
 * Check how large a document is for the AI.
 *
 * Free and instant. Used to decide whether to offer the optimization option at all —
 * on a small document, reducing context only loses detail.
 */
export async function getContextPreview(id: string): Promise<ContextPreview> {
  return apiRequest<ContextPreview>(`/documents/${id}/context-preview`);
}

/** Check whether the server can make AI requests at all. */
export async function getAIStatus(): Promise<AIStatus> {
  return apiRequest<AIStatus>("/ai/status");
}

/**
 * Ask a question about a document.
 *
 * Unlike summary/explanation/extraction, this is never cached: two different questions
 * about the same document share nothing to cache against, and the endpoint does not
 * support multi-turn conversation yet. Every call is a fresh, billed request.
 */
export async function askQuestion(
  id: string,
  question: string,
  topK?: number,
): Promise<AskResponse> {
  const body: AskRequest = {
    question,
    ...(topK !== undefined ? { top_k: topK } : {}),
  };
  return apiRequest<AskResponse>(`/documents/${id}/ask`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Turn a document into a structured, multi-record table.
 *
 * `input` is either `{ instruction }` — a plain-language description of the table
 * wanted, from which the server works out the fields and returns them as `schema_used` —
 * or `{ schema }`, naming the fields explicitly. The union makes sending both
 * unrepresentable; the server rejects that combination, but it should never get that far.
 *
 * Returns every matching record the document describes, not just the first. Cached per
 * resolved schema; pass `refresh` to regenerate.
 */
export async function structureDocument(
  id: string,
  input: StructureInput,
  refresh = false,
): Promise<StructureResponse> {
  const body: StructureRequest = { ...input, refresh };
  return apiRequest<StructureResponse>(`/documents/${id}/structure`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Export an already-generated `/structure` result as a downloadable file.
 *
 * Never calls the AI model: `schema` must match a previously generated `/structure`
 * result exactly (field order does not matter) or the server responds 404. The response
 * is a binary file rather than JSON, so this goes through {@link apiRequestBlob} instead
 * of {@link apiRequest}.
 */
export async function exportStructure(
  id: string,
  schema: StructureSchema,
  format: StructureExportFormat,
): Promise<DownloadedFile> {
  return apiRequestBlob(`/documents/${id}/structure/export?format=${format}`, {
    method: "POST",
    body: JSON.stringify({ schema }),
  });
}
