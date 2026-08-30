/**
 * The one button.
 *
 * Three variants, and each says something different about the action: `primary`
 * is the thing the panel exists to do and there is at most one per view;
 * `secondary` is a real alternative alongside it; `subtle` is a control that
 * belongs to a row or a cell and should not compete with either.
 *
 * Deliberately not styled per call site. Every button in the workflows used to
 * carry its own copy of the same eight classes, which is how a disabled state
 * ends up looking different in two places.
 */

type ButtonVariant = "primary" | "secondary" | "subtle";
type ButtonSize = "sm" | "md";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

// `active:` gives the press itself a frame of feedback. A button that changes
// only on hover feels unresponsive on the click, and on a touch screen — where
// there is no hover at all — it never acknowledges anything.
const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-accent-ink hover:bg-accent-hover active:bg-accent-hover border border-transparent",
  secondary:
    "border-line-strong text-ink hover:bg-sunken hover:border-ink-subtle active:bg-sunken border",
  subtle: "border-line text-ink-muted hover:text-ink hover:bg-sunken active:bg-sunken border",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
};

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  type = "button",
  ...props
}: ButtonProps): React.JSX.Element {
  return (
    <button
      // Defaulted because a bare <button> inside a form submits it, which has
      // caused accidental submissions in this codebase's forms before.
      type={type}
      className={`rounded-control focus-visible:outline-accent inline-flex items-center justify-center font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1 disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...props}
    />
  );
}
