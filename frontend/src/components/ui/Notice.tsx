/**
 * Something the user should know that is not a failure.
 *
 * Carries the warnings the server returns with a result — content left out, a
 * source that could not be confirmed. These are surfaced rather than hidden:
 * a result with an unstated caveat is worse than one that admits it.
 */

interface NoticeProps {
  children: React.ReactNode;
}

export function Notice({ children }: NoticeProps): React.JSX.Element {
  return (
    <div className="border-caution/40 bg-caution-soft text-caution rounded-container border px-3 py-2 text-xs">
      {children}
    </div>
  );
}
