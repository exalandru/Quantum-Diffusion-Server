/**
 * A modal surface, for the controls a catalogue row must offer without carrying.
 *
 * Not `<dialog>`: `showModal()` is imperative, which means the open state would
 * live in the DOM and in React at once, and the two disagreeing is the classic
 * way a dialog gets stuck open. React stays the single owner of whether the
 * surface exists; a caller mounts this component and unmounts it, exactly as
 * before.
 *
 * The behaviour underneath is `react-aria-components` rather than hand-rolled.
 * That closes the one thing the previous implementation openly did not do: it
 * documented that focus was *not* trapped, because a correct tab cycle is more
 * than a `keydown` listener and claiming it without one is worse than not
 * claiming it. RAC contains focus for real, marks the rest of the page
 * `aria-hidden` while the surface is up, and restores focus on the way out.
 *
 * Two deliberate departures from RAC's defaults, both to keep this a swap of
 * mechanism and not of appearance or of contract:
 *
 *  - The overlay is portalled *into the caller's own subtree*, not into
 *    `document.body`. RAC portals to the body by default, which would move the
 *    dialog out of the row that opened it — and the catalogue's tests scope
 *    their queries to that row (`within(row).getByRole("dialog")`). Rendering
 *    where the caller rendered is also simply what this component did before.
 *    `.modal-backdrop` is `position: fixed`, so it still covers the viewport
 *    from there.
 *  - `<Modal>` is `display: contents` and the `.modal` class sits on `<Dialog>`.
 *    RAC's `Modal` is a second box between the backdrop and the dialog; letting
 *    it style itself would have inserted a block into a flex line that
 *    `styles.css` sizes and centres. `display: contents` removes the box and
 *    leaves the cascade addressing the same two elements it always did.
 *
 * Escape and the backdrop press are RAC's now. The backdrop press got stricter
 * in the process, and in the direction this file already wanted: it needs the
 * press *and* the release outside the surface, so a text selection dragged out
 * of the dialog no longer dismisses it.
 *
 * Focus restoration stays this component's own, as it was before RAC. RAC does
 * restore focus, but only when the overlay it owns is still mounted as it
 * closes — it restores from a `FocusScope` teardown. Every caller here mounts
 * the dialog conditionally and unmounts it to close, so the scope and the
 * surface disappear in the same commit and the restore lands on `<body>`.
 * Measured: with restoration left to RAC, `document.activeElement` after
 * Escape was `<body>` rather than the button that opened the dialog, which is
 * the regression `Models.test.tsx`'s "hands focus back to what opened it"
 * exists to catch. The effect below is the original one, kept for that reason.
 *
 * The exception is `portalToBody`, for a caller that sits inside a glass
 * surface. `.modal-backdrop` is `position: fixed`, and a fixed element's
 * containing block is not the viewport when an ancestor carries `transform`,
 * `filter` or `backdrop-filter` — it is that ancestor. The playground's
 * composer is `backdrop-filter: blur(…)`, so a dialog opened from the gear
 * inside it was laid out *within the composer*: measured at 1440×820, the
 * surface landed at y=479 with its foot 115px below the window. Callers in that
 * position ask for the body, and give up the `within(row)` scoping they were
 * not using anyway.
 */
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { UNSAFE_PortalProvider } from "react-aria";
import {
  Dialog,
  Heading,
  Modal as RACModal,
  ModalOverlay,
} from "react-aria-components";

export function Modal({
  title,
  subtitle,
  onClose,
  className,
  portalToBody,
  bare,
  children,
}: {
  title: string;
  /** The model's source identity, so the dialog says which one it is editing. */
  subtitle?: string;
  onClose: () => void;
  /** Extra class on the surface, for a dialog that needs another size. */
  className?: string;
  /** Escape a `backdrop-filter` ancestor: see the note above. */
  portalToBody?: boolean;
  /**
   * Draw no head and no body padding: the surface is the caller's content.
   *
   * `title` is still required and still names the dialog — through `aria-label`
   * rather than a visible heading. Escape, the backdrop press and the focus
   * round-trip are unaffected: this hides chrome, not behaviour.
   */
  bare?: boolean;
  children: ReactNode;
}) {
  // The portal target is an empty element rendered right here, so the overlay
  // lands in this component's own position in the tree. It is state rather than
  // a ref because the provider needs to re-render once the node exists — a ref
  // would still be null on the first pass and RAC would fall back to the body.
  const [host, setHost] = useState<HTMLElement | null>(null);
  const getContainer = useCallback(
    () => (portalToBody ? document.body : host),
    [host, portalToBody],
  );

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    return () => {
      // Only if it is still on screen: the opener can be a button the closing
      // action itself removed.
      if (opener && opener.isConnected) opener.focus();
    };
  }, []);

  return (
    <div ref={setHost}>
      {(host !== null || portalToBody === true) && (
        <UNSAFE_PortalProvider getContainer={getContainer}>
          <ModalOverlay
            isOpen
            isDismissable
            onOpenChange={(open) => {
              if (!open) onClose();
            }}
            className="modal-backdrop"
          >
            <RACModal style={{ display: "contents" }}>
              <Dialog
                className={className ? `modal ${className}` : "modal"}
                // A bare surface renders no `Heading slot="title"`, so RAC has
                // nothing to derive the dialog's accessible name from. The title
                // still names it, as a label rather than as visible text: an
                // unnamed dialog is announced as "dialog" and nothing else.
                aria-label={bare ? title : undefined}
              >
                {/* Chrome by default, none when the caller draws its own. The
                    image viewer is the one surface whose subject *is* the
                    picture: a titled panel around it frames a frame, and the
                    prompt it used as a title competed with the image for the
                    top of the screen. What `bare` drops is the head and the
                    body's padding — never Escape, the backdrop press or the
                    focus round-trip, which are the reason this component exists
                    and which a caller drawing its own chrome must not have to
                    reimplement. */}
                {bare ? (
                  children
                ) : (
                  <>
                    <div className="modal-head">
                      <Heading slot="title" level={3} className="modal-title">
                        {title}
                      </Heading>
                      {subtitle && <code className="modal-subtitle">{subtitle}</code>}
                      <button type="button" className="small modal-close" onClick={onClose}>
                        Close
                      </button>
                    </div>
                    <div className="modal-body">{children}</div>
                  </>
                )}
              </Dialog>
            </RACModal>
          </ModalOverlay>
        </UNSAFE_PortalProvider>
      )}
    </div>
  );
}
