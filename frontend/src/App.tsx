/**
 * Routes.
 *
 * A document's workspace is a URL — `/d/{id}/{tab}` — so it can be linked,
 * bookmarked, refreshed and walked with the back button. Everything renders
 * inside one layout route, which keeps the library mounted across navigation.
 *
 * Two redirects rather than error pages: `/d/{id}` picks the default tab, and
 * anything unrecognised returns to the library. Both replace rather than push,
 * so Back never lands on a URL that immediately redirects again.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { DEFAULT_TAB } from "@/lib/workspaceTabs";
import { DocumentWorkspace } from "@/routes/DocumentWorkspace";
import { ShellLayout } from "@/routes/ShellLayout";

export default function App(): React.JSX.Element {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<ShellLayout />}>
          <Route index element={<EmptyWorkspace />} />
          <Route path="d/:documentId" element={<Navigate to={DEFAULT_TAB} replace />} />
          <Route path="d/:documentId/:tab" element={<DocumentWorkspace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

/**
 * Shown when no document is open.
 *
 * States what to do next rather than decorating the space. On a small screen
 * the library is behind the drawer, so the wording has to name that.
 */
function EmptyWorkspace(): React.JSX.Element {
  return (
    <div className="flex h-full items-center justify-center px-6 py-16">
      <div className="max-w-sm text-center">
        <p className="text-ink text-sm font-medium">No document open</p>
        <p className="text-ink-muted mt-1 text-sm">
          Choose a document from the library to summarize it, ask questions about it, or
          turn it into structured data.
        </p>
        <p className="text-ink-subtle mt-3 text-xs md:hidden">
          Open the library with the menu button above.
        </p>
      </div>
    </div>
  );
}
