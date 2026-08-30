/**
 * Display formatting for document metadata.
 *
 * Shared rather than copied: file sizes now appear in the upload tray, the
 * library and the workspace header, and three independent implementations is
 * how "244.0 KB" in one place becomes "0.2 MB" in another.
 */

/** `812 B`, `244.0 KB`, `3.7 MB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * An ISO timestamp in the viewer's own locale and timezone.
 *
 * Degrades to a label rather than "Invalid Date" — a malformed timestamp is a
 * reason to show less, not to print something that looks like a bug.
 */
export function formatDate(isoTimestamp: string): string {
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) return "unknown date";
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
