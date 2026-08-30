/**
 * The workspace tabs, and the mapping between them and the URL.
 *
 * Kept apart from the components so the route, the tab bar and the fallback
 * logic all agree on one list. A tab that exists in the UI but not here would
 * be unreachable by link; one here but not in the UI would 404 silently.
 *
 * The slugs are the values themselves — `/d/{id}/summary` — so a URL says what
 * it shows without a translation table to keep in step.
 */

/** A view within one document's workspace. */
export type WorkspaceTab =
  | "summary"
  | "explanation"
  | "extraction"
  | "ask"
  | "structure"
  | "convert";

/** Where a document opens, and where an unrecognised tab falls back to. */
export const DEFAULT_TAB: WorkspaceTab = "summary";

export const WORKSPACE_TABS: readonly WorkspaceTab[] = [
  "summary",
  "explanation",
  "extraction",
  "ask",
  "structure",
  "convert",
];

/** Narrow a URL segment, which may be anything a person typed. */
export function isWorkspaceTab(value: string | undefined): value is WorkspaceTab {
  return value !== undefined && (WORKSPACE_TABS as readonly string[]).includes(value);
}

/** The workspace URL for a document and tab. */
export function workspacePath(documentId: string, tab: WorkspaceTab = DEFAULT_TAB): string {
  return `/d/${documentId}/${tab}`;
}
