/**
 * A modal surface, for the controls a catalogue row must offer without carrying.
 *
 * Not `<dialog>`: `showModal()` is imperative, which means the open state would
 * live in the DOM and in React at once, and the two disagreeing is the classic
 * way a dialog gets stuck open. A rendered element with `role="dialog"` and
 * `aria-modal` keeps React the single owner of whether it exists.
 *
 * What it does owe a keyboard user, it does: the surface takes focus on open,
 * Escape closes it, and focus returns to whatever opened it. Focus is not
 * *trapped* — that would need a full tab-cycle implementation, and claiming it
 * without one is worse than not claiming it. The surface is the last thing in
 * the panel's DOM order, so tabbing out lands in the shell rather than behind
 * the backdrop.
 */
import { useEffect, useId, useRef, type ReactNode } from "react";

export function Modal({
  title,
  subtitle,
  onClose,
  className,
  children,
}: {
  title: string;
  /** The model's source identity, so the dialog says which one it is editing. */
  subtitle?: string;
  onClose: () => void;
  /** Extra class on the surface, for a dialog that needs another size. */
  className?: string;
  children: ReactNode;
}) {
  const titleId = useId();
  const surface = useRef<HTMLDivElement>(null);
  // Read through a ref so a caller's inline arrow does not re-run the effect —
  // which would re-take focus on every render of the panel behind it.
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    surface.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      // Only if it is still on screen: the opener can be a button the closing
      // action itself removed.
      if (opener && opener.isConnected) opener.focus();
    };
  }, []);

  return (
    <div
      className="modal-backdrop"
      // `mousedown` rather than `click`: a drag that starts inside the surface
      // and ends on the backdrop is a text selection, not a dismissal.
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close.current();
      }}
    >
      <div
        className={className ? `modal ${className}` : "modal"}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        ref={surface}
      >
        <div className="modal-head">
          <h3 className="modal-title" id={titleId}>
            {title}
          </h3>
          {subtitle && <code className="modal-subtitle">{subtitle}</code>}
          <button type="button" className="small modal-close" onClick={() => close.current()}>
            Close
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
