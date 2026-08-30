/**
 * What a panel shows before it has anything to show.
 *
 * Says what the panel will do and what it costs, rather than filling the space
 * with an illustration. Every workflow here is a paid AI request, so "nothing
 * yet" is the normal opening state, not an edge case.
 */

interface EmptyStateProps {
  children: React.ReactNode;
}

export function EmptyState({ children }: EmptyStateProps): React.JSX.Element {
  return (
    <div className="border-line bg-sunken/40 rounded-container border border-dashed px-6 py-10">
      {/* Measured rather than full-bleed. Centred text set across a wide panel
          is read by jumping back to a different left edge on every line, which
          is exactly the wrong treatment for the sentence explaining what a
          panel does. */}
      <p className="text-ink-muted mx-auto max-w-prose text-center text-sm leading-relaxed text-balance">
        {children}
      </p>
    </div>
  );
}
