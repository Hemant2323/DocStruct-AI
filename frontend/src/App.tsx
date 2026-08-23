/**
 * Application shell.
 *
 * Milestone 1 renders a single page proving the frontend/backend link works. Routing
 * arrives in Milestone 5 when there is more than one screen to route between — adding a
 * router now would be complexity with nothing to show for it.
 */

import { BackendStatus } from "@/components/BackendStatus";
import { DocumentList } from "@/components/DocumentList";

export default function App(): React.JSX.Element {
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
        <DocumentList />

        <footer className="pt-2 text-center text-xs text-slate-500 dark:text-slate-500">
          Milestone 1 — foundation. Upload, extraction, summaries and Q&amp;A
          arrive in later milestones.
        </footer>
      </main>
    </div>
  );
}
