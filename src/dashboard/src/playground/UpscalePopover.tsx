import { useCallback, useRef } from "react";

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
  /** The toolbar button that opened this, so pressing it again closes. */
  trigger?: React.RefObject<HTMLElement | null>;
}) {
  const panel = useRef<HTMLDivElement>(null);
  useDismissable(true, panel, useCallback(() => onClose(), [onClose]), trigger);

  const selected = upscalers.find((entry) => entry.id === choice.model) ?? upscalers[0];
  if (selected === undefined) return null;

  return (
    <div ref={panel} className="pg-popover pg-upscale-pop" role="dialog" aria-label="Upscale options">
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
    </div>
  );
}
