/** Typed wrappers around the document endpoints. */

import { apiRequest } from "@/api/client";
import type {
  DocumentListResponse,
  DocumentSummary,
  ProcessingStatus,
} from "@/types/api";

export interface ListDocumentsParams {
  limit?: number;
  offset?: number;
  status?: ProcessingStatus;
}

/** Fetch a page of documents, newest first. */
export async function listDocuments(
  params: ListDocumentsParams = {},
): Promise<DocumentListResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  if (params.status !== undefined) query.set("status", params.status);

  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return apiRequest<DocumentListResponse>(`/documents${suffix}`);
}

/** Fetch a single document's metadata and status. */
export async function getDocument(id: string): Promise<DocumentSummary> {
  return apiRequest<DocumentSummary>(`/documents/${id}`);
}

/** Delete a document and its stored content. */
export async function deleteDocument(id: string): Promise<void> {
  await apiRequest<void>(`/documents/${id}`, { method: "DELETE" });
}
