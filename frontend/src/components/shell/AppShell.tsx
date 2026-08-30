/**
 * The application frame: a top bar, a persistent sidebar, and a workspace.
 *
 * The shape follows the way the product is actually used. A document library is
 * something you return to constantly — picking one, watching another finish
 * processing, uploading a third — so it stays visible rather than being a step
 * you scroll past. The workspace holds one document at a time, which is what
 * gives "the thing I am working on" somewhere to live.
 *
 * Below `md` there is not room for both, so the sidebar becomes a drawer over
 * the workspace: the same markup, positioned differently, rather than a second
 * navigation built for small screens.
 *
 * Structure only. No workflow is aware it is inside this.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ThemeToggle } from "@/components/ThemeToggle";

interface AppShellProps {
  /** Upload control and document library. */
  sidebar: React.ReactNode;
  /**
   * Pinned below the library. Normally renders nothing — it carries the
   * connection banner, which only appears when the connection is broken.
   */
  sidebarFooter?: React.ReactNode;
  /** The open document's workflows. */
  children: React.ReactNode;
  /**
   * Bumped when a document is chosen, so the drawer closes on small screens.
   * A drawer left open over the thing it was used to select would have to be
   * dismissed manually every time.
   */
  closeDrawerSignal?: number;
}

export function AppShell({
  sidebar,
  sidebarFooter,
  children,
  closeDrawerSignal,
}: AppShellProps): React.JSX.Element {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (closeDrawerSignal === undefined) return;
    setDrawerOpen(false);
  }, [closeDrawerSignal]);

  useEffect(() => {
    if (!drawerOpen) return;

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setDrawerOpen(false);
        // Escape must put focus somewhere deliberate. Left alone it stays on an
        // element that is now display:none, and the next Tab restarts from the
        // top of the document.
        toggleRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen]);

  // Opening the drawer moves focus into it, so the first Tab continues inside
  // the library rather than behind it.
  useEffect(() => {
    if (!drawerOpen) return;
    sidebarRef.current?.querySelector<HTMLElement>("button, [href], input")?.focus();
  }, [drawerOpen]);

  return (
    <div className="bg-canvas text-ink flex h-full flex-col">
      {/* Off-screen until focused. Without it, reaching the workspace by
          keyboard means tabbing past the upload control and every row in the
          library — which can be fifty documents. */}
      <a
        href="#workspace"
        className="bg-surface text-ink border-accent focus:rounded-container sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:border focus:px-3 focus:py-2 focus:text-sm focus:font-medium"
      >
        Skip to document
      </a>

      <header className="border-line bg-surface flex h-14 shrink-0 items-center gap-3 border-b px-4">
        <button
          ref={toggleRef}
          type="button"
          onClick={() => setDrawerOpen((open) => !open)}
          aria-expanded={drawerOpen}
          aria-controls="document-library"
          aria-label="Toggle document library"
          className="border-line text-ink-muted hover:text-ink rounded-control focus-visible:outline-accent border p-1.5 transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1 md:hidden"
        >
          {/* Functional: the only way to reach the library on a small screen. */}
          <svg viewBox="0 0 16 16" aria-hidden="true" className="h-4 w-4 fill-current">
            <rect x="2" y="3.25" width="12" height="1.5" rx="0.75" />
            <rect x="2" y="7.25" width="12" height="1.5" rx="0.75" />
            <rect x="2" y="11.25" width="12" height="1.5" rx="0.75" />
          </svg>
        </button>

        <span className="text-ink text-sm font-semibold tracking-tight">DocStruct AI</span>
        <span className="text-ink-subtle hidden text-xs sm:inline">
          Document intelligence
        </span>

        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        {drawerOpen && (
          <button
            type="button"
            aria-label="Close document library"
            onClick={closeDrawer}
            className="absolute inset-0 z-20 bg-black/30 md:hidden"
          />
        )}

        <aside
          ref={sidebarRef}
          id="document-library"
          aria-label="Document library"
          // Wider on a large screen: the library's rows carry a filename, a
          // metadata line and a state, and at 288px long filenames truncated
          // more often than they needed to.
          className={`border-line bg-surface z-30 flex w-72 shrink-0 flex-col border-r lg:w-80 ${
            drawerOpen
              ? "absolute inset-y-0 left-0 md:relative"
              : "hidden md:flex"
          }`}
        >
          {/* A column, not a scroll container. The upload control stays put at
              the top and the library scrolls inside itself — scrolling the two
              together pushed "Add documents" off the screen exactly when a full
              library made it hardest to get back to. */}
          <div className="flex min-h-0 flex-1 flex-col">{sidebar}</div>
          {/* No border or padding here: the footer is usually nothing at all,
              and a wrapper that styles an empty child draws a rule across the
              sidebar for no reason. Whatever renders brings its own frame. */}
          {sidebarFooter !== undefined && <div className="shrink-0">{sidebarFooter}</div>}
        </aside>

        {/* The routed view renders here, header included: a `sticky` header
            inside this scroll container pins correctly, so the shell does not
            need a separate slot for it. */}
        <main
          id="workspace"
          tabIndex={-1}
          className="flex min-w-0 flex-1 flex-col overflow-y-auto"
        >
          {children}
        </main>
      </div>
    </div>
  );
}
