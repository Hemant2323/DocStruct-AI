/** Typed wrappers around the document endpoints. */

import { apiRequest } from "@/api/client";
import type {
  ContentResponse,
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

/**
 * Upload a file for ingestion.
 *
 * Resolves as soon as the server accepts the file (HTTP 202). Processing happens in the
 * background, so the returned document has `status: "pending"` — poll {@link getDocument}
 * until the status becomes `ready` or `failed`.
 */
export async function uploadDocument(file: File): Promise<DocumentSummary> {
  const form = new FormData();
  form.append("file", file);

  return apiRequest<DocumentSummary>("/documents", {
    method: "POST",
    body: form,
  });
}

/** Fetch a page of a document's extracted content blocks. */
export async function getDocumentContent(
  id: string,
  params: { limit?: number; offset?: number } = {},
): Promise<ContentResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));

  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return apiRequest<ContentResponse>(`/documents/${id}/content${suffix}`);
}
