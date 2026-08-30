/**
 * A failed request, next to the control that failed.
 *
 * Inline rather than a global toast: an error here belongs to one panel, and
 * moving it elsewhere would separate it from the thing to retry.
 *
 * The request ID is shown whenever the server supplied one — it is the key to
 * the matching server log line, and the only useful thing a user can quote when
 * reporting a 500 whose message is deliberately generic.
 */

interface ErrorBlockProps {
  message: string;
  requestId?: string | null;
}

export function ErrorBlock({ message, requestId }: ErrorBlockProps): React.JSX.Element {
  return (
    <div
      role="alert"
      className="border-critical/40 bg-critical-soft rounded-container border px-3 py-2.5"
    >
      <p className="text-critical text-sm">{message}</p>
      {requestId !== undefined && requestId !== null && (
        <p className="text-ink-subtle mt-1.5 font-mono text-xs">Reference: {requestId}</p>
      )}
    </div>
  );
}
