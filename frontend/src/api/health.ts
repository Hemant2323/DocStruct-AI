/** Typed wrappers around the health endpoints. */

import { apiRequest } from "@/api/client";
import type { HealthResponse, ReadinessResponse } from "@/types/api";

/** Liveness: is the backend process up? */
export async function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}

/** Readiness: are the backend's dependencies reachable? */
export async function getReadiness(): Promise<ReadinessResponse> {
  return apiRequest<ReadinessResponse>("/ready");
}
