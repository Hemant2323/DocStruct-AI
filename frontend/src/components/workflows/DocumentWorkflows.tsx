/**
 * The five things you can do with an open document.
 *
 * This component owns only what is common to all of them: the tab bar, the
 * "AI is not configured" notice, and the shared insight state. Each workflow
 * owns its own panel — which is what makes them independently readable, and
 * what lets Ask and Structure keep request shapes that have nothing to do with
 * Summary's.
 *
 * Which tab is showing comes from the route, not from state here, so a workflow
 * can be linked to directly.
 */

import { Notice } from "@/components/ui/Notice";
import { Tabs, tabId, tabPanelId } from "@/components/ui/Tabs";
import type { TabDefinition } from "@/components/ui/Tabs";
import { AskPanel } from "@/components/workflows/AskPanel";
import { ConvertPanel } from "@/components/workflows/ConvertPanel";
import { ExplainPanel } from "@/components/workflows/ExplainPanel";
import { ExtractPanel } from "@/components/workflows/ExtractPanel";
import { StructurePanel } from "@/components/workflows/StructurePanel";
import { SummaryPanel } from "@/components/workflows/SummaryPanel";
import { useInsights } from "@/components/workflows/useInsights";
import type { WorkspaceTab } from "@/lib/workspaceTabs";

/** Ties each tab to the single panel it controls. */
const TAB_ID_PREFIX = "workflow";

interface DocumentWorkflowsProps {
  documentId: string;
  /** Which view to show. Owned by the route so a tab can be linked to. */
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
}

export function DocumentWorkflows({
  documentId,
  activeTab,
  onTabChange,
}: DocumentWorkflowsProps): React.JSX.Element {
  const insights = useInsights(documentId);
  const { results, aiUnavailable } = insights;

  // A dot marks a tab whose result already exists: these cost money to produce,
  // so returning to one is worth distinguishing from regenerating it.
  const tabs: TabDefinition<WorkspaceTab>[] = [
    {
      id: "summary",
      label: "Summary",
      hint: "A short description of what this document says.",
      hasResult: results.summary !== undefined,
    },
    {
      id: "explanation",
      label: "Explain simply",
      hint: "The same content, written for someone new to the subject.",
      hasResult: results.explanation !== undefined,
    },
    {
      id: "extraction",
      label: "Extract information",
      hint: "Key facts pulled out as structured data.",
      hasResult: results.extraction !== undefined,
    },
    {
      id: "ask",
      label: "Ask a question",
      hint: "Ask about this document and get an answer with sources.",
    },
    {
      id: "structure",
      label: "Structure document",
      hint: "Define the fields you want and get them back as structured data.",
    },
    {
      id: "convert",
      label: "Convert",
      hint: "Build a new, organized document out of this one, with every change audited.",
    },
  ];

  return (
    <section className="px-6 py-5">
      {/* The tab bar names each workflow visually; this gives the section the
          heading level that keeps the document outline unbroken between the
          filename above (h1) and the headings inside a result (h3). */}
      <h2 className="sr-only">Document workflows</h2>

      {aiUnavailable && (
        <div className="mb-4">
          <Notice>
            AI features are not configured on this server. An Anthropic API key needs to
            be set in the backend’s <code className="font-mono">.env</code> file.
          </Notice>
        </div>
      )}

      <Tabs
        tabs={tabs}
        active={activeTab}
        onChange={onTabChange}
        label="Document workflows"
        idPrefix={TAB_ID_PREFIX}
      />

      <div
        id={tabPanelId(TAB_ID_PREFIX)}
        role="tabpanel"
        aria-labelledby={tabId(TAB_ID_PREFIX, activeTab)}
        // Focusable because a panel's content is often plain text with nothing
        // to land on; without this, arrowing to a tab leaves its panel
        // unreachable by keyboard.
        tabIndex={0}
        className="border-line focus-visible:outline-accent border-t pt-5 focus-visible:outline-2 focus-visible:-outline-offset-2"
      >
        {activeTab === "summary" && <SummaryPanel insights={insights} />}
        {activeTab === "explanation" && <ExplainPanel insights={insights} />}
        {activeTab === "extraction" && <ExtractPanel insights={insights} />}
        {activeTab === "ask" && (
          <AskPanel documentId={documentId} aiUnavailable={aiUnavailable} />
        )}
        {activeTab === "structure" && (
          <StructurePanel documentId={documentId} aiUnavailable={aiUnavailable} />
        )}
        {activeTab === "convert" && (
          <ConvertPanel documentId={documentId} aiUnavailable={aiUnavailable} />
        )}
      </div>
    </section>
  );
}
