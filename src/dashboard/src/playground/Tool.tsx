import type { ReactNode, Ref } from "react";

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
  expanded,
  onClick,
  buttonRef,
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
  /** Set when the button opens a panel, so it can say whether it is open. */
  expanded?: boolean;
  /**
   * The underlying button, for a caller that opens a panel.
   *
   * A dismiss-on-outside-press handler has to know the trigger, or pressing it
   * while the panel is open closes on `mousedown` and reopens on `click` — a
   * toggle that never toggles off.
   */
  buttonRef?: Ref<HTMLButtonElement>;
  onClick?: () => void;
  /** The icon's shape elements. */
  children: ReactNode;
}) {
  return (
    <button
      ref={buttonRef}
      type="button"
      className={danger ? "small danger pg-tool" : "small pg-tool"}
      data-tip={native ? undefined : tip}
      title={native ? tip : undefined}
      aria-label={label ?? tip}
      aria-expanded={expanded}
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
