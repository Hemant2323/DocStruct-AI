/**
 * Shows where an AI result came from.
 *
 * The distinction this component exists to make visible is **verified versus not**. A
 * confirmed source shows the place in the file it came from; one that could not be
 * confirmed says so plainly. Nothing is ever displayed as a location unless the server
 * resolved it from stored content, so a user can trust what they see here.
 */

import type { Citation } from "@/types/api";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({
  citations,
}: CitationListProps): React.JSX.Element | null {
  if (citations.length === 0) {
    return null;
  }

  const verified = citations.filter((citation) => citation.is_verified).length;

  return (
    <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-700">
      <h4 className="mb-2 text-sm font-medium text-slate-700 dark:text-slate-300">
        Sources
        <span className="ml-2 font-normal text-slate-500 dark:text-slate-400">
          ({verified} of {citations.length} confirmed)
        </span>
      </h4>

      <ul className="space-y-2">
        {citations.map((citation, index) => (
          <li
            key={`${citation.location_label}-${index}`}
            className="rounded-md border border-slate-200 bg-slate-50 p-2.5 text-xs dark:border-slate-700 dark:bg-slate-900"
          >
            <div className="mb-1 flex items-center gap-2">
              {citation.is_verified ? (
                <span className="rounded bg-emerald-100 px-2 py-0.5 font-mono font-medium text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
                  {citation.location_label}
                </span>
              ) : (
                <span
                  title="The AI referred to something we could not match to a specific place in your document. No location is shown rather than a guessed one."
                  className="rounded bg-amber-100 px-2 py-0.5 font-medium text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                >
                  source not verified
                </span>
              )}
            </div>
            <p className="text-slate-600 italic dark:text-slate-400">
              “{citation.quoted_text}”
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
