/**
 * User-facing copy for document processing states.
 *
 * The backend's status values (`pending`, `processing`, `ready`, `failed`) are precise
 * but not friendly — "pending" does not tell a person what is happening or whether they
 * need to do anything. This module is the single place that turns each state into plain
 * language, so every part of the UI describes a state the same way.
 */

import type { ProcessingStatus } from "@/types/api";

/** How prominent a state should look. Maps to colours at the component level. */
export type StatusTone = "neutral" | "info" | "success" | "danger";

export interface StatusCopy {
  /** Short label for a badge. */
  label: string;
  /** One sentence explaining what is happening, in plain language. */
  description: string;
  tone: StatusTone;
  /** True while the document is still moving through the pipeline. */
  inProgress: boolean;
}

export const DOCUMENT_STATUS_COPY: Record<ProcessingStatus, StatusCopy> = {
  pending: {
    label: "Waiting",
    description: "Uploaded and queued. Processing will start shortly.",
    tone: "neutral",
    inProgress: true,
  },
  processing: {
    label: "Processing",
    description: "Reading the document and pulling out its contents.",
    tone: "info",
    inProgress: true,
  },
  ready: {
    label: "Ready",
    description: "Finished. This document is ready to use.",
    tone: "success",
    inProgress: false,
  },
  failed: {
    label: "Failed",
    description: "We could not read this document.",
    tone: "danger",
    inProgress: false,
  },
};

/**
 * Return the display copy for a status.
 *
 * Falls back to a safe "Unknown" rather than crashing if the backend ever introduces a
 * status this build has not been taught about — an older frontend against a newer API
 * should degrade, not white-screen.
 */
export function statusCopy(status: ProcessingStatus): StatusCopy {
  return (
    DOCUMENT_STATUS_COPY[status] ?? {
      label: "Unknown",
      description: "This document is in a state this app does not recognise.",
      tone: "neutral",
      inProgress: false,
    }
  );
}

/**
 * Build the full message shown for a document.
 *
 * For a failed document the backend's `error_message` explains the specific cause
 * (encrypted PDF, corrupt file, and so on), so it is preferred over the generic text.
 * The generic sentence is used only when no reason was recorded.
 */
export function statusMessage(
  status: ProcessingStatus,
  errorMessage: string | null,
): string {
  const copy = statusCopy(status);
  if (status === "failed" && errorMessage) {
    return errorMessage;
  }
  return copy.description;
}
