/**
 * Holds the theme preference and keeps the document in step with it.
 *
 * Two things happen here that are easy to get subtly wrong. The resolved theme
 * is applied on every change, including the first, so the attribute set before
 * paint by the inline script in `index.html` stays authoritative rather than
 * being contradicted a frame later. And while the preference is "system", the
 * OS setting is *watched*, not merely read once — a machine that switches to
 * dark at sunset should take the app with it.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { ThemeContext, type ThemeContextValue } from "@/lib/theme-context";
import {
  type ResolvedTheme,
  type ThemePreference,
  applyTheme,
  readStoredPreference,
  resolveTheme,
  storePreference,
  watchSystemTheme,
} from "@/lib/theme";

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps): React.JSX.Element {
  // Read straight from storage on the first render rather than starting on a
  // default and correcting in an effect, which would flash the wrong theme.
  const [preference, setPreferenceState] = useState<ThemePreference>(() =>
    readStoredPreference(),
  );
  const [theme, setTheme] = useState<ResolvedTheme>(() => resolveTheme(readStoredPreference()));

  useEffect(() => {
    const resolved = resolveTheme(preference);
    setTheme(resolved);
    applyTheme(resolved);
  }, [preference]);

  useEffect(() => {
    // Only meaningful while deferring to the OS; an explicit choice should not
    // be overridden by the machine changing its mind.
    if (preference !== "system") return;

    return watchSystemTheme((next) => {
      setTheme(next);
      applyTheme(next);
    });
  }, [preference]);

  const setPreference = useCallback((next: ThemePreference): void => {
    storePreference(next);
    setPreferenceState(next);
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ preference, theme, setPreference }),
    [preference, theme, setPreference],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
