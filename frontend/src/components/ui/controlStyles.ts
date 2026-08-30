/**
 * The shared look of a text control.
 *
 * A constant rather than an `Input`/`TextArea`/`Select` trio: the three elements
 * need different props and behaviour but the *same* border, background, radius,
 * focus ring and disabled treatment, and that shared part is the only thing that
 * kept drifting. Ask's question box and Structure's field rows had each grown
 * their own copy, already differing in radius.
 *
 * Sizing and width stay at the call site, where they depend on the layout.
 */
export const FIELD_CLASS =
  "border-line-strong bg-surface text-ink rounded-control focus-visible:outline-accent " +
  "border p-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-1 " +
  "disabled:cursor-not-allowed disabled:opacity-50";
