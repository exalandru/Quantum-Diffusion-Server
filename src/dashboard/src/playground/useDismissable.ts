import { useEffect } from "react";

/**
 * Close an open popover on a press outside it, or on Escape.
 *
 * Shared rather than repeated: `mousedown` and not `click` is the load-bearing
 * detail — with `click`, a text selection begun inside the panel and released
 * outside it counts as a press outside and dismisses the panel mid-drag.
 */
export function useDismissable(
  open: boolean,
  container: React.RefObject<HTMLElement | null>,
  close: () => void,
  /**
   * The control that opens the panel, when it sits outside `container`.
   *
   * Pressing it must not count as a press outside, or the panel closes on
   * `mousedown` and the trigger's own handler reopens it on `click` — the
   * button then never closes what it opened. Callers whose ref already wraps
   * both, as the composer's does, need not pass this.
   */
  trigger?: React.RefObject<HTMLElement | null>,
) {
  useEffect(() => {
    if (!open) return;
    function onPointer(event: MouseEvent) {
      const target = event.target as Node;
      if (container.current?.contains(target)) return;
      if (trigger?.current?.contains(target)) return;
      close();
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") close();
    }
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, container, close, trigger]);
}
