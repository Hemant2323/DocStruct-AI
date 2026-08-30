/**
 * The theme context and its hook.
 *
 * Split from the provider component so each file exports one kind of thing:
 * Fast Refresh only preserves state reliably for modules that export
 * components alone, and mixing a hook in causes the whole module to remount on
 * edit.
 */

import { createContext, useContext } from "react";

import type { ResolvedTheme, ThemePreference } from "@/lib/theme";

export interface ThemeContextValue {
  /** What the user chose. May be "system". */
  preference: ThemePreference;
  /** What is displayed right now, after "system" has been resolved. */
  theme: ResolvedTheme;
  setPreference: (preference: ThemePreference) => void;
}

export const ThemeContext = createContext<ThemeContextValue | null>(null);

/**
 * Read the current theme.
 *
 * @throws When used outside {@link ThemeProvider} — a silent default would let
 *   a mis-mounted component quietly disagree with the rest of the app about
 *   which theme is in force.
 */
export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext);
  if (value === null) {
    throw new Error("useTheme must be used inside a ThemeProvider");
  }
  return value;
}
