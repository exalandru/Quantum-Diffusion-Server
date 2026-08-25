import { useCallback, useEffect, useRef } from "react";
import { Dialog, Popover } from "react-aria-components";

import type { Upscaler } from "../types";
import { useDismissable } from "./useDismissable";

export type UpscaleChoice = { model: string; scale: number };

/**
 * Factor and model for one enlargement.
 *
 * Rendered as a sibling of the image toolbar rather than inside it: a
 * `role="toolbar"` holds peer controls, not a dialog, and keeping it out means
 * the toolbar still answers with exactly four buttons whether this is open or
 * shut — which is what the feed's test asserts.
 *
 * Positioned by `react-aria-components`' `Popover`, which portals to the body.
 * Step 1 deliberately chose a bare `Dialog` positioned by the stylesheet
 * instead — absolute, `bottom: calc(100% + …)`, against `.pg-image-actions` —
 * and that decision is reversed here, because the mechanism it preserved is the
 * bug: `.pg-feed` and `.pg-gallery` are `overflow-y: auto`, and a scroll
 * container clips anything its descendants paint outside its box. A panel hung
 * above a tile near the top of the view was cut in half and its remainder
 * painted further down the page. No amount of `bottom`/`left` tuning fixes
 * that; a panel that cannot leave its scroll container cannot be placed.
 *
 * So the panel now lives in a body portal — nothing above it clips — and RAC
 * measures the trigger to place it: `placement="top"` with flipping left on, so
 * it opens above the toolbar where there is room and below it where there is
 * not, and `containerPadding` keeps it off the viewport edge.
 *
 * `isNonModal` is the deliberate half of adopting `Popover`. A modal popover
 * would add an underlay, lock page scroll and mark the rest of the document
 * inert for a two-field panel, and it would take `role="dialog"` for the
 * positioned wrapper. Non-modal keeps the page live, and the inner `Dialog` is
 * what carries `role="dialog"` and the accessible name — RAC skips its own
 * dialog role when the popover already contains one.
 *
 * Dismissal stays `useDismissable`, and RAC's `onOpenChange` is deliberately
 * *not* wired: RAC positions this panel and owns nothing else about it.
 *
 * That is not a preference. A non-modal RAC popover asks `useOverlay` to close
 * on focus leaving the panel (`shouldCloseOnBlur: true`, not overridable
 * through `Popover`'s props), and the inner `Dialog` takes focus when it opens.
 * Pressing the trigger a second time therefore blurs the panel, RAC closes it,
 * and the trigger's own `click` — which is the toggle — reopens what the press
 * had just shut: measured, the panel never toggles off, which is exactly what
 * `GenerationFeed`'s "closes the panel from the button that opened it" exists
 * to catch. `useDismissable` has no such collision because it exempts the
 * trigger by design.
 *
 * The one thing RAC's close would have given is the effect below: a panel
 * placed in a body portal is positioned once, against the trigger where it was,
 * so scrolling the feed or the gallery under it would leave it hanging over a
 * picture that has moved. Closing on any scroll is the same answer RAC's own
 * `useCloseOnScroll` gives, applied here where the trigger exemption lives.
 */
export function UpscalePopover({
  upscalers,
  choice,
  onChoose,
  onSubmit,
  onClose,
  trigger,
}: {
  upscalers: Upscaler[];
  choice: UpscaleChoice;
  onChoose: (choice: UpscaleChoice) => void;
  onSubmit: () => void;
  onClose: () => void;
  /** The toolbar button that opened this: it anchors the panel and toggles it. */
  trigger: React.RefObject<HTMLElement | null>;
}) {
  const panel = useRef<HTMLDivElement>(null);
  useDismissable(true, panel, useCallback(() => onClose(), [onClose]), trigger);

  useEffect(() => {
    // Capture: a scroll event does not bubble, so a listener on `document`
    // hears `.pg-feed` and `.pg-gallery` only on the way down.
    const close = () => onClose();
    document.addEventListener("scroll", close, { capture: true });
    return () => document.removeEventListener("scroll", close, { capture: true });
  }, [onClose]);

  const selected = upscalers.find((entry) => entry.id === choice.model) ?? upscalers[0];
  if (selected === undefined) return null;

  return (
    <Popover
      ref={panel}
      isOpen
      isNonModal
      triggerRef={trigger}
      placement="top"
      offset={10}
      containerPadding={12}
      className="pg-popover"
    >
      <Dialog className="pg-popover-fields" aria-label="Upscale options">
        <div className="pg-field">
          <span className="pg-field-label">Factor</span>
          <div className="pg-seg" role="group" aria-label="Factor">
            {selected.scales.map((scale) => (
              <button
                key={scale}
                type="button"
                aria-pressed={choice.scale === scale}
                onClick={() => onChoose({ ...choice, scale })}
              >
                ×{scale}
              </button>
            ))}
          </div>
          {/* Said plainly because the shape of the thing invites the opposite
              guess: the network only knows one factor, and a smaller one is that
              result resampled down. Half the size is not half the wait. */}
          <p className="note">×2 is a ×4 resampled down — the same wait either way.</p>
        </div>

        <div className="pg-field">
          <span className="pg-field-label">Model</span>
          <div className="pg-seg" role="group" aria-label="Model">
            {upscalers.map((entry) => (
              <button
                key={entry.id}
                type="button"
                aria-pressed={choice.model === entry.id}
                onClick={() => onChoose({ ...choice, model: entry.id })}
              >
                {entry.name}
              </button>
            ))}
          </div>
          {!selected.downloaded && (
            // The weights download inside the run, holding the queue. Without
            // this the first click just looks like a hang.
            <p className="note">First use downloads {selected.sizeMb} MB.</p>
          )}
        </div>

        <div className="actions">
          <button type="button" className="primary" onClick={onSubmit}>
            Upscale
          </button>
        </div>
      </Dialog>
    </Popover>
  );
}
