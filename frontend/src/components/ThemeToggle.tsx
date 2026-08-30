/**
 * The theme control: light, system, dark.
 *
 * Three explicit states rather than a two-way switch, because "follow the OS"
 * is a real choice and a binary toggle silently discards it — a user who has
 * never touched this should be able to see that the app is deferring, and to
 * get back to deferring after trying a theme.
 *
 * Rendered as a radiogroup so the arrow keys work and screen readers announce
 * it as one control with three options, which is what it is.
 */

import { useCallback, useRef } from "react";

import type { ThemePreference } from "@/lib/theme";
import { useTheme } from "@/lib/theme-context";

const OPTIONS: { value: ThemePreference; label: string; title: string }[] = [
  { value: "light", label: "Light", title: "Always use the light theme" },
  { value: "system", label: "Auto", title: "Follow the operating system" },
  { value: "dark", label: "Dark", title: "Always use the dark theme" },
];

export function ThemeToggle(): React.JSX.Element {
  const { preference, setPreference } = useTheme();
  const groupRef = useRef<HTMLDivElement>(null);

  /**
   * A radiogroup is one tab stop whose options are reached with the arrow keys.
   * Three separate tab stops would both contradict the role and put two extra
   * stops in front of everything else in the header.
   */
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>): void => {
      const OFFSETS: Record<string, number> = {
        ArrowRight: 1,
        ArrowDown: 1,
        ArrowLeft: -1,
        ArrowUp: -1,
      };
      const offset = OFFSETS[event.key];
      if (offset === undefined) return;

      const current = OPTIONS.findIndex((option) => option.value === preference);
      if (current === -1) return;

      const next = OPTIONS[(current + offset + OPTIONS.length) % OPTIONS.length];
      if (next === undefined) return;

      event.preventDefault();
      setPreference(next.value);
      groupRef.current
        ?.querySelector<HTMLButtonElement>(`[data-theme-option="${next.value}"]`)
        ?.focus();
    },
    [preference, setPreference],
  );

  return (
    <div
      ref={groupRef}
      role="radiogroup"
      aria-label="Theme"
      onKeyDown={onKeyDown}
      className="border-line rounded-container inline-flex border p-0.5"
    >
      {OPTIONS.map((option) => {
        const selected = preference === option.value;
        return (
          <button
            key={option.value}
            data-theme-option={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            title={option.title}
            onClick={() => setPreference(option.value)}
            className={`rounded-control focus-visible:outline-accent px-2 py-1 text-xs font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1 ${
              selected
                ? "bg-sunken text-ink"
                : "text-ink-subtle hover:text-ink"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
