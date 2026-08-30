/**
 * One document's workspace, addressed by URL.
 *
 * The document is fetched by id rather than handed over from the library,
 * because the URL has to work on its own: pasted into a new tab, restored after
 * a refresh, or reached with the back button, there may be no list in memory to
 * take it from. Fetching also means the header reflects the document as it is
 * now — a document that finished processing since it was listed shows as ready.
 */

import { useCallback, useEffect, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "@/api/client";
import { getDocument } from "@/api/documents";
import { DocumentContent } from "@/components/DocumentContent";
import { DocumentHeader } from "@/components/shell/DocumentHeader";
import { DocumentWorkflows } from "@/components/workflows/DocumentWorkflows";
import { DEFAULT_TAB, isWorkspaceTab, workspacePath } from "@/lib/workspaceTabs";
import type { WorkspaceTab } from "@/lib/workspaceTabs";
import type { DocumentSummary } from "@/types/api";

type LoadState =
  | { kind: "loading" }
  | { kind: "loaded"; document: DocumentSummary }
  | { kind: "missing" }
  | { kind: "error"; message: string; requestId: string | null };

export function DocumentWorkspace(): React.JSX.Element {
  const { documentId, tab } = useParams<{ documentId: string; tab: string }>();
  const navigate = useNavigate();
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    if (documentId === undefined) return;

    let cancelled = false;
    setState({ kind: "loading" });

    const load = async (): Promise<void> => {
      try {
        const document = await getDocument(documentId);
        if (!cancelled) setState({ kind: "loaded", document });
      } catch (error) {
        if (cancelled) return;
        // A link to a document that no longer exists is a normal thing to
        // follow, not a failure to report as one.
        if (error instanceof ApiError && error.status === 404) {
          setState({ kind: "missing" });
          return;
        }
        setState({
          kind: "error",
          message:
            error instanceof ApiError ? error.message : "Could not open this document.",
          requestId: error instanceof ApiError ? error.requestId : null,
        });
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const handleTabChange = useCallback(
    (next: WorkspaceTab) => {
      if (documentId === undefined) return;
      navigate(workspacePath(documentId, next));
    },
    [documentId, navigate],
  );

  const handleClose = useCallback(() => navigate("/"), [navigate]);

  if (documentId === undefined) {
    return <Navigate to="/" replace />;
  }

  // An unrecognised tab is corrected rather than shown as an error: the
  // document is real and the intent is obvious. `replace` keeps the bad URL out
  // of history, so Back does not return to it.
  if (!isWorkspaceTab(tab)) {
    return <Navigate to={workspacePath(documentId, DEFAULT_TAB)} replace />;
  }

  if (state.kind === "loading") {
    return (
      <div className="text-ink-subtle px-6 py-8 text-sm">Opening document…</div>
    );
  }

  if (state.kind === "missing") {
    return (
      <WorkspaceNotice
        title="Document not found"
        detail="It may have been deleted, or the link may be wrong."
      />
    );
  }

  if (state.kind === "error") {
    return (
      <WorkspaceNotice
        title="Could not open this document"
        detail={state.message}
        requestId={state.requestId}
      />
    );
  }

  return (
    <>
      <DocumentHeader document={state.document} onClose={handleClose} />
      <DocumentWorkflows
        documentId={state.document.id}
        activeTab={tab}
        onTabChange={handleTabChange}
      />
      <div className="border-line border-t">
        <DocumentContent documentId={state.document.id} />
      </div>
    </>
  );
}

function WorkspaceNotice({
  title,
  detail,
  requestId,
}: {
  title: string;
  detail: string;
  requestId?: string | null;
}): React.JSX.Element {
  return (
    <div className="flex h-full items-center justify-center px-6 py-16">
      <div className="max-w-sm text-center">
        <p className="text-ink text-sm font-medium">{title}</p>
        <p className="text-ink-muted mt-1 text-sm">{detail}</p>
        {requestId !== undefined && requestId !== null && (
          <p className="text-ink-subtle mt-2 font-mono text-xs">Reference: {requestId}</p>
        )}
      </div>
    </div>
  );
}
