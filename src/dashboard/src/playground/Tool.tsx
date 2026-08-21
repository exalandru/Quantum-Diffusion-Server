import type { ReactNode } from "react";

/**
 * One icon-only toolbar button.
 *
 * Shared by the feed's per-image toolbar and the sidebar's session actions, so
 * there is one icon button in this app rather than two that drift apart.
 *
 * `aria-label` always carries the name for assistive technology. The visible
 * tooltip has two forms, and the choice is not cosmetic: `data-tip` feeds the
 * CSS tooltip, which is styled like the rest of the app but is drawn *outside*
 * the button and so is clipped by any scroll container above it — which is
 * exactly what `.pg-sidebar` is. `native` asks the browser for its own tooltip
 * instead, which no ancestor can clip.
 */
export function Tool({
  tip,
  label,
  danger,
  disabled,
  native,
  onClick,
  children,
}: {
  /** The tooltip text, and the accessible name unless `label` overrides it. */
  tip: string;
  /**
   * The accessible name, when it should say more than the tooltip does — a
   * sidebar full of "Delete" buttons needs to name *which* session each deletes.
   */
  label?: string;
  danger?: boolean;
  disabled?: boolean;
  /** Use the browser's own tooltip: for a button inside a scroll container. */
  native?: boolean;
  onClick?: () => void;
  /** The icon's shape elements. */
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={danger ? "small danger pg-tool" : "small pg-tool"}
      data-tip={native ? undefined : tip}
      title={native ? tip : undefined}
      aria-label={label ?? tip}
      disabled={disabled}
      onClick={onClick}
    >
      <svg
        viewBox="0 0 24 24"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {children}
      </svg>
    </button>
  );
}
