/**
 * Application shell.
 *
 * One page: upload, the document list with live status, and — for whichever document is
 * selected — its AI analysis and its extracted content. Routing arrives in Milestone 5,
 * when there is more than one screen to route between.
 */

import { useCallback, useState } from "react";

import { BackendStatus } from "@/components/BackendStatus";
import { DocumentContent } from "@/components/DocumentContent";
import { DocumentInsights } from "@/components/DocumentInsights";
import { DocumentList } from "@/components/DocumentList";
import { DocumentUpload } from "@/components/DocumentUpload";
import type { DocumentSummary } from "@/types/api";

export default function App(): React.JSX.Element {
  // Bumped after each upload so the list refreshes immediately rather than waiting for
  // its next poll.
  const [refreshToken, setRefreshToken] = useState(0);
  const [selected, setSelected] = useState<DocumentSummary | null>(null);

  const handleUploaded = useCallback(() => {
    setRefreshToken((token) => token + 1);
  }, []);

  return (
    <div className="min-h-full bg-slate-50 dark:bg-slate-900">
      <header className="border-b border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
        <div className="mx-auto max-w-4xl px-6 py-6">
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            DocStruct AI
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            AI-powered document intelligence and structuring platform
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-6 px-6 py-8">
        <BackendStatus />
        <DocumentUpload onUploaded={handleUploaded} />
        <DocumentList
          refreshToken={refreshToken}
          selectedId={selected?.id ?? null}
          onSelect={setSelected}
        />

        {selected !== null && (
          <>
            <DocumentInsights
              documentId={selected.id}
              filename={selected.original_filename}
            />
            <DocumentContent
              documentId={selected.id}
              filename={selected.original_filename}
              onClose={() => setSelected(null)}
            />
          </>
        )}

        <footer className="pt-2 text-center text-xs text-slate-500 dark:text-slate-500">
          Milestone 3 — AI understanding. Conversational question answering arrives in a
          later milestone.
        </footer>
      </main>
    </div>
  );
}
