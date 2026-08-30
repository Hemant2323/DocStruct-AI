/**
 * Theme preference: reading it, resolving it, and putting it on the document.
 *
 * Kept as plain functions rather than folded into the provider so the awkward
 * parts — a stored value that may be anything, a browser that throws on
 * `localStorage`, an OS preference that changes while the app is open — can be
 * reasoned about and exercised without rendering a component.
 *
 * The distinction that matters throughout: a **preference** may be "system",
 * but a **resolved theme** is only ever "light" or "dark". The DOM carries the
 * resolved value, so nothing downstream has to interpret the attribute.
 */

/** What the user chose. "system" defers to the OS. */
export type ThemePreference = "light" | "dark" | "system";

/** What is actually displayed, after "system" has been resolved. */
export type ResolvedTheme = "light" | "dark";

/** Where the preference is remembered. Namespaced so it cannot collide. */
export const THEME_STORAGE_KEY = "docstruct.theme";

/** The attribute `index.css` keys its `dark:` variant off. */
export const THEME_ATTRIBUTE = "data-theme";

const PREFERENCES: readonly ThemePreference[] = ["light", "dark", "system"];

const DARK_QUERY = "(prefers-color-scheme: dark)";

/** Narrow an unknown stored value. Anything unrecognised is not a preference. */
export function isThemePreference(value: unknown): value is ThemePreference {
  return typeof value === "string" && (PREFERENCES as readonly string[]).includes(value);
}

/**
 * The stored preference, or "system" when there is none.
 *
 * Every failure path lands on "system": no stored value, a value written by an
 * older version that no longer means anything, or a browser that refuses
 * storage entirely (private windows, blocked site data). Following the OS is
 * the right thing to do when we do not know what the user wants.
 */
export function readStoredPreference(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

/**
 * Remember a preference, or forget it when the user returns to "system".
 *
 * Storing "system" as a literal would freeze today's OS setting into a choice
 * the user never made; removing the key keeps it genuinely deferred.
 *
 * A write that fails is not worth interrupting anyone over — the theme still
 * applies for this session, it just will not be remembered.
 */
export function storePreference(preference: ThemePreference): void {
  try {
    if (preference === "system") {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, preference);
    }
  } catch {
    // Storage unavailable. Ignored deliberately: see above.
  }
}

/** What the OS currently asks for. Defaults to light where unknowable. */
export function systemTheme(): ResolvedTheme {
  try {
    return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

/** Turn a preference into the theme to actually display. */
export function resolveTheme(preference: ThemePreference): ResolvedTheme {
  return preference === "system" ? systemTheme() : preference;
}

/** Put the resolved theme on `<html>`, where the CSS variant reads it. */
export function applyTheme(theme: ResolvedTheme): void {
  document.documentElement.setAttribute(THEME_ATTRIBUTE, theme);
  // Lets form controls, scrollbars and the like match without extra styling.
  document.documentElement.style.colorScheme = theme;
}

/**
 * Watch the OS setting, calling back when it changes.
 *
 * Live rather than read-once: someone whose machine switches to dark at sunset
 * should see the app follow, not find it stale until a reload.
 *
 * @returns A function that stops watching.
 */
export function watchSystemTheme(onChange: (theme: ResolvedTheme) => void): () => void {
  let query: MediaQueryList;
  try {
    query = window.matchMedia(DARK_QUERY);
  } catch {
    return () => {};
  }

  const handle = (event: MediaQueryListEvent): void => {
    onChange(event.matches ? "dark" : "light");
  };

  query.addEventListener("change", handle);
  return () => query.removeEventListener("change", handle);
}
