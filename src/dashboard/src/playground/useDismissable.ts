import { useEffect, useRef } from "react";
import { useInteractOutside } from "react-aria";

/**
 * Close an open popover on a press outside it, or on Escape.
 *
 * The outside-press half is `react-aria`'s `useInteractOutside` rather than a
 * hand-rolled listener. The load-bearing detail is unchanged and is in fact
 * enforced more strictly than before: a text selection begun inside the panel
 * and released outside it must not dismiss the panel mid-drag. The old code got
 * that by listening for `mousedown` instead of `click`, so a drag out of the
 * panel was ignored because its *press* was inside. `useInteractOutside`
 * requires the press and the release to both be outside, which also covers the
 * case the old rule missed — a selection dragged *into* the panel from outside,
 * which used to dismiss it on the way in.
 *
 * Escape stays a `document` listener here rather than moving to react-aria's
 * `useOverlay`. `useOverlay` delivers Escape through props spread onto the
 * overlay element, so it only fires while focus is inside the panel; these
 * popovers are opened by a button that keeps focus, and Escape has to work from
 * there.
 *
 * It listens in the *capture* phase, and that is load-bearing since the upscale
 * panel became a `react-aria-components` `Popover`. RAC's own overlay binds
 * Escape through `useKeyboard` on the popover element and stops the event
 * there; the panel's inner `Dialog` takes focus when it opens, so a bubbling
 * listener on `document` never saw the key and the panel stopped closing on
 * Escape. Capture runs before the target's own handlers, so this hears it
 * either way.
 */
export function useDismissable(
  open: boolean,
  container: React.RefObject<HTMLElement | null>,
  close: () => void,
  /**
   * The control that opens the panel, when it sits outside `container`.
   *
   * Pressing it must not count as a press outside, or the panel closes on the
   * press and the trigger's own handler reopens it on `click` — the button then
   * never closes what it opened. Callers whose ref already wraps both, as the
   * composer's does, need not pass this.
   *
   * `useInteractOutside` takes a single ref, so the exemption is applied here:
   * it reports the interaction and this decides whether it counts.
   */
  trigger?: React.RefObject<HTMLElement | null>,
) {
  // Read through refs so the identity of a caller's inline arrow does not
  // matter; `useInteractOutside` re-subscribes on ref or disabled changes only.
  const latest = useRef({ close, trigger });
  latest.current = { close, trigger };

  useInteractOutside({
    ref: container,
    isDisabled: !open,
    onInteractOutside: (event) => {
      const { close: onClose, trigger: triggerRef } = latest.current;
      const target = event.target as Node | null;
      if (target && triggerRef?.current?.contains(target)) return;
      onClose();
    },
  });

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") latest.current.close();
    }
    document.addEventListener("keydown", onKey, { capture: true });
    return () => document.removeEventListener("keydown", onKey, { capture: true });
  }, [open]);
}
