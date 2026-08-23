/**
 * The single HTTP client for the DocStruct AI API.
 *
 * Per CLAUDE.md, components never call `fetch` directly — everything goes through here.
 * That keeps error handling, base-URL resolution, and response parsing in one place
 * instead of being re-implemented (slightly differently) in every component.
 */

import type { ApiErrorBody } from "@/types/api";

/**
 * Base URL for API calls.
 *
 * Empty by default, which produces same-origin relative requests (`/api/health`). In
 * development Vite proxies those to the backend, so no CORS round-trip is involved. Set
 * `VITE_API_BASE_URL` to point at a backend on a different origin.
 */
const API_BASE_URL: string = import.meta.env["VITE_API_BASE_URL"] ?? "";

/** Path prefix for every API route. Must match the backend's `API_PREFIX`. */
const API_PREFIX = "/api";

/**
 * An error raised by an API call.
 *
 * Carries the backend's machine-readable `code` so callers can branch on the kind of
 * failure rather than pattern-matching a message string.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;
  /**
   * Correlation ID for the failed request, when the server supplied one. Worth showing
   * to the user on an unexpected error — it is the key to the matching server log entry.
   */
  readonly requestId: string | null;

  constructor(
    status: number,
    code: string,
    message: string,
    details: Record<string, unknown> = {},
    requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }

  /** True when the request never reached the server (backend down, DNS, offline). */
  get isNetworkError(): boolean {
    return this.status === 0;
  }
}

/** Narrows an unknown parsed body to the API's error envelope. */
function isApiErrorBody(body: unknown): body is ApiErrorBody {
  if (typeof body !== "object" || body === null || !("error" in body)) {
    return false;
  }
  const { error } = body as { error: unknown };
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    "message" in error
  );
}

/**
 * Perform an API request and decode the JSON response.
 *
 * Every failure path produces an {@link ApiError} — a non-2xx status, an unreachable
 * server, or a malformed body. Nothing is swallowed and no call resolves with a
 * half-valid value.
 *
 * @param path Path below the API prefix, e.g. `"/health"`.
 * @param init Standard `fetch` options.
 * @throws {ApiError} On any failure.
 */
export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE_URL}${API_PREFIX}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch (cause) {
    // fetch rejects only when the request could not be made at all.
    throw new ApiError(
      0,
      "network_error",
      "Could not reach the DocStruct AI backend. Is it running?",
      { cause: String(cause) },
    );
  }

  // Present on every response; the backend exposes it via CORS so JS can read it.
  const headerRequestId = response.headers.get("X-Request-ID");

  // 204 No Content has no body to parse (the DELETE endpoint returns this).
  if (response.status === 204) {
    return undefined as T;
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    if (!response.ok) {
      throw new ApiError(
        response.status,
        "invalid_response",
        `Request failed with status ${response.status}`,
        {},
        headerRequestId,
      );
    }
    throw new ApiError(
      response.status,
      "invalid_response",
      "The server returned a response that was not valid JSON",
      {},
      headerRequestId,
    );
  }

  if (!response.ok) {
    if (isApiErrorBody(body)) {
      throw new ApiError(
        response.status,
        body.error.code,
        body.error.message,
        body.error.details ?? {},
        body.error.request_id ?? headerRequestId,
      );
    }
    throw new ApiError(
      response.status,
      "unknown_error",
      `Request failed with status ${response.status}`,
      {},
      headerRequestId,
    );
  }

  return body as T;
}
