/** Typed wrappers around the AI analysis endpoints. */

import { apiRequest } from "@/api/client";
import type {
  AIStatus,
  AnalysisOptions,
  ContextPreview,
  Insight,
  InsightListResponse,
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
