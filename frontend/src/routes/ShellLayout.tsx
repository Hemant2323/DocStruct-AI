/**
 * The layout every route renders inside.
 *
 * The sidebar lives here rather than in each route so the library keeps its
 * scroll position, its poll timer and its loaded state while you move between
 * documents — remounting it on navigation would restart all three and make the
 * list flicker on every click.
 *
 * Which document is highlighted is read from the URL rather than held in state,
 * so the highlight, the workspace and the address bar cannot disagree.
 */

import { useCallback, useState } from "react";
import { Outlet, useMatch, useNavigate } from "react-router-dom";

import { BackendStatus } from "@/components/BackendStatus";
import { DocumentList } from "@/components/DocumentList";
import { DocumentUpload } from "@/components/DocumentUpload";
import { AppShell } from "@/components/shell/AppShell";
import { workspacePath } from "@/lib/workspaceTabs";
import type { DocumentSummary } from "@/types/api";

export function ShellLayout(): React.JSX.Element {
  const navigate = useNavigate();
  // Matched loosely: the highlight should hold on every tab of a document, so
  // it must not depend on which tab is open.
  const match = useMatch("/d/:documentId/*");
  const selectedId = match?.params.documentId ?? null;

  // Bumped after each upload so the list refreshes at once rather than waiting
  // for its next poll.
  const [refreshToken, setRefreshToken] = useState(0);
  // Bumped on selection so the shell dismisses its drawer on small screens.
  const [selectionCount, setSelectionCount] = useState(0);

  const handleUploaded = useCallback(() => {
    setRefreshToken((token) => token + 1);
  }, []);

  const handleSelect = useCallback(
    (document: DocumentSummary | null) => {
      if (document === null) {
        navigate("/");
        return;
      }
      setSelectionCount((count) => count + 1);
      navigate(workspacePath(document.id));
    },
    [navigate],
  );

  return (
    <AppShell
      closeDrawerSignal={selectionCount}
      sidebar={
        <>
          <DocumentUpload onUploaded={handleUploaded} />
          <DocumentList
            refreshToken={refreshToken}
            selectedId={selectedId}
            onSelect={handleSelect}
          />
        </>
      }
      sidebarFooter={<BackendStatus />}
    >
      <Outlet />
    </AppShell>
  );
}
