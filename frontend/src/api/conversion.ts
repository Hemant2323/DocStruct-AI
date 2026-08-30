/** Typed wrappers around the document conversion endpoints. */

import { apiRequest, apiRequestBlob, type DownloadedFile } from "@/api/client";
import type {
  ConversionExportFormat,
  ConversionListResponse,
  ConversionRequest,
  ConversionResponse,
} from "@/types/api";

/** The server's limit on an instruction, mirrored so the box can say so before sending. */
export const MAX_INSTRUCTION_LENGTH = 2000;

/**
 * Build a new, organized document out of an existing one.
 *
 * POST because it costs several model calls and stores a result. Unlike
 * `/summary` and friends there is **no conversion-level cache**: the same
 * instruction legitimately yields a different document, so every call produces
 * a new stored conversion. `refresh` controls only whether the *extraction*
 * behind it is reused.
 */
export async function convertDocument(
  id: string,
  instruction: string,
  refresh = false,
): Promise<ConversionResponse> {
  const body: ConversionRequest = { instruction, refresh };
  return apiRequest<ConversionResponse>(`/documents/${id}/convert`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** List the conversions already produced for a document. Free — never calls the AI. */
export async function listConversions(id: string): Promise<ConversionListResponse> {
  return apiRequest<ConversionListResponse>(`/documents/${id}/conversions`);
}

/**
 * Fetch a stored conversion by id.
 *
 * Read-only and free: the server returns the result exactly as it was produced,
 * with nothing regenerated. This is what makes a past conversion something a
 * user can return to rather than a number that changes each time they look.
 */
export async function getConversion(
  id: string,
  conversionId: string,
): Promise<ConversionResponse> {
  return apiRequest<ConversionResponse>(`/documents/${id}/conversions/${conversionId}`);
}

/**
 * Download a stored conversion as a file.
 *
 * A `GET` that reformats the stored result — no AI call and no re-extraction, so
 * the bytes are the same every time. The response is binary, so this goes
 * through {@link apiRequestBlob} rather than {@link apiRequest}.
 */
export async function exportConversion(
  id: string,
  conversionId: string,
  format: ConversionExportFormat,
): Promise<DownloadedFile> {
  return apiRequestBlob(
    `/documents/${id}/conversions/${conversionId}/export?format=${format}`,
  );
}
