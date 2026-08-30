/**
 * The workspace tab bar.
 *
 * An underline marks the active tab rather than a filled pill: the tabs sit
 * directly above the panel they switch, so the marker should connect them, not
 * float as a separate object.
 *
 * A tab may carry a dot showing it already holds a result — useful because
 * those results cost money to produce and are worth returning to rather than
 * regenerating.
 *
 * Keyboard behaviour follows the ARIA tabs pattern: the bar is a single tab
 * stop and the arrow keys move between tabs. Six tabs would otherwise be six
 * stops standing between the sidebar and the panel a keyboard user is actually
 * heading for.
 */

import { useCallback, useRef } from "react";

import { StatusDot } from "@/components/ui/StatusDot";

export interface TabDefinition<T extends string> {
  id: T;
  label: string;
  /** Shown on hover; the panel repeats it for anyone not using a pointer. */
  hint?: string;
  /** Marks a tab that already has a generated result. */
  hasResult?: boolean;
}

interface TabsProps<T extends string> {
  tabs: readonly TabDefinition<T>[];
  active: T;
  onChange: (id: T) => void;
  label: string;
  /**
   * Prefix for the generated `id`s that tie each tab to its panel. Must match
   * what the panel container uses.
   */
  idPrefix: string;
}

export function tabId(prefix: string, id: string): string {
  return `${prefix}-tab-${id}`;
}

export function tabPanelId(prefix: string): string {
  return `${prefix}-panel`;
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
  label,
  idPrefix,
}: TabsProps<T>): React.JSX.Element {
  const listRef = useRef<HTMLDivElement>(null);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>): void => {
      const OFFSETS: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1 };
      const offset = OFFSETS[event.key];

      let nextIndex: number | null = null;
      const currentIndex = tabs.findIndex((tab) => tab.id === active);

      if (offset !== undefined && currentIndex !== -1) {
        // Wraps, as the pattern specifies — reaching the end and continuing
        // should not simply stop dead.
        nextIndex = (currentIndex + offset + tabs.length) % tabs.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = tabs.length - 1;
      }

      if (nextIndex === null) return;
      const next = tabs[nextIndex];
      if (next === undefined) return;

      event.preventDefault();
      onChange(next.id);
      // The newly selected tab is the only one in the tab order, so focus has
      // to be moved to it explicitly or it lands nowhere.
      listRef.current
        ?.querySelector<HTMLButtonElement>(`#${CSS.escape(tabId(idPrefix, next.id))}`)
        ?.focus();
    },
    [active, idPrefix, onChange, tabs],
  );

  return (
    // Scrolls rather than wrapping or widening the page: six tabs do not fit a
    // phone, and the body must never scroll sideways.
    <div
      ref={listRef}
      role="tablist"
      aria-label={label}
      onKeyDown={onKeyDown}
      className="border-line -mb-px flex gap-4 overflow-x-auto border-b"
    >
      {tabs.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            id={tabId(idPrefix, tab.id)}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={tabPanelId(idPrefix)}
            // Roving tabindex: one stop for the whole bar.
            tabIndex={selected ? 0 : -1}
            title={tab.hint}
            onClick={() => onChange(tab.id)}
            className={`focus-visible:outline-accent -mb-px flex items-center gap-1.5 border-b-2 px-0.5 pb-2 text-sm font-medium whitespace-nowrap transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-2 ${
              selected
                ? "border-accent text-ink"
                : "text-ink-subtle hover:text-ink border-transparent"
            }`}
          >
            {tab.label}
            {tab.hasResult === true && (
              <StatusDot tone="positive" label="has a saved result" />
            )}
          </button>
        );
      })}
    </div>
  );
}
