/**
 * What can be done to one generated image, wherever it is shown.
 *
 * Lifted out of `GenerationFeed` when the gallery arrived. The four actions are
 * the same four in both views and they mean the same thing — refine and vary
 * read the *root's* settings, upscale names a file the server already owns,
 * delete costs a picture — so a second copy of this markup would be four icons
 * and one popover kept in step by hand. The submission logic itself is neither
 * copy's: it stays in `PlaygroundApp`, and both views hand it in.
 *
 * The open panel and the chosen factor are the caller's state, not this
 * component's: only one panel may be open across a whole view, and the choice is
 * remembered across images so enlarging ten of them does not cost twenty clicks
 * re-picking the same factor and model.
 */
import { useRef, useState, type Dispatch, type SetStateAction } from "react";

import type { PlaygroundGeneration, Upscaler } from "../types";
import type { GroupImage } from "./groups";
import { Tool } from "./Tool";
import { UpscalePopover, type UpscaleChoice } from "./UpscalePopover";

export type ImageToolsState = {
  /** The image whose upscale panel is open, by URL. One at a time. */
  open: string | null;
  setOpen: Dispatch<SetStateAction<string | null>>;
  choice: UpscaleChoice;
  setChoice: Dispatch<SetStateAction<UpscaleChoice>>;
  /**
   * The open panel's trigger. One ref is enough because one panel is open at a
   * time, and it is the open one whose trigger must not read as "outside".
   */
  trigger: React.RefObject<HTMLButtonElement | null>;
};

/** The state above, owned by the view that renders the tools. */
export function useImageTools(): ImageToolsState {
  const [open, setOpen] = useState<string | null>(null);
  const [choice, setChoice] = useState<UpscaleChoice>({ model: "", scale: 2 });
  const trigger = useRef<HTMLButtonElement>(null);
  return { open, setOpen, choice, setChoice, trigger };
}

export function ImageTools({
  image,
  root,
  busy,
  upscalers,
  tools,
  onRefine,
  onVariation,
  onUpscale,
  onDeleteImage,
}: {
  image: GroupImage;
  /** The generation that opened the lineage: it owns prompt, model, size, steps. */
  root: PlaygroundGeneration;
  /** A submission is in flight: the generating actions wait for it. */
  busy: boolean;
  /**
   * The upscaler catalogue, or an empty list.
   *
   * Empty means the control stays disabled — the same fail-closed rule the
   * composer's drop zone and the advanced fields follow: an offer the server
   * may refuse is worse than no offer.
   */
  upscalers: Upscaler[];
  tools: ImageToolsState;
  onRefine: (root: PlaygroundGeneration, image: { url: string; seed: number }) => void;
  onVariation: (root: PlaygroundGeneration) => void;
  onUpscale: (
    root: PlaygroundGeneration,
    image: { url: string; seed: number },
    choice: UpscaleChoice,
  ) => void;
  onDeleteImage: (url: string) => void;
}) {
  const { open, setOpen, choice, setChoice, trigger } = tools;
  return (
    <div className="pg-image-actions">
      <div className="pg-image-tools" role="toolbar" aria-label="Image actions">
        <Tool
          tip="Refine"
          disabled={busy}
          // The bare image, not the group's annotated copy: `kind`, `size` and
          // `model` are the views' bookkeeping for the badge and the viewer's
          // footer, and leaking them would make them contract.
          onClick={() => onRefine(root, { url: image.url, seed: image.seed })}
        >
          <path d="M12 3.5l1.5 4.6 4.6 1.5-4.6 1.5L12 15.7l-1.5-4.6L5.9 9.6l4.6-1.5z" />
          <path d="M17.8 15.4l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
        </Tool>
        <Tool tip="New variation" disabled={busy} onClick={() => onVariation(root)}>
          <path d="M4 7h4l8 10h4M4 17h4l2.8-3.5M13.2 10.5 16 7h4M17 4l3 3-3 3M17 14l3 3-3 3" />
        </Tool>
        <Tool
          tip={upscalers.length === 0 ? "Upscale unavailable" : "Upscale"}
          disabled={busy || upscalers.length === 0}
          expanded={open === image.url}
          buttonRef={open === image.url ? trigger : undefined}
          onClick={() =>
            setOpen((current) => {
              if (current === image.url) return null;
              // Default to the first catalogue entry rather than a hardcoded
              // id: the catalogue is the server's.
              if (!upscalers.some((entry) => entry.id === choice.model)) {
                setChoice((held) => ({ ...held, model: upscalers[0]?.id ?? "" }));
              }
              return image.url;
            })
          }
        >
          <rect x="3.5" y="14" width="6.5" height="6.5" rx="1.2" />
          <rect x="11.5" y="3.5" width="9" height="9" rx="1.6" />
          <path d="M10.2 13.8 12.8 11.2M12.8 13.7v-2.5h-2.5" />
        </Tool>
        <Tool
          tip="Delete image"
          danger
          onClick={() => {
            // The convention for a destructive one-click action here —
            // SessionList does the same for whole sessions.
            if (window.confirm("Delete this image?")) onDeleteImage(image.url);
          }}
        >
          <path d="M4 7h16M9 7V5h6v2M6.5 7l.8 12h9.4l.8-12M10 11v5M14 11v5" />
        </Tool>
      </div>
      {open === image.url && (
        <UpscalePopover
          upscalers={upscalers}
          choice={choice}
          onChoose={setChoice}
          trigger={trigger}
          onClose={() => setOpen(null)}
          onSubmit={() => {
            setOpen(null);
            onUpscale(root, { url: image.url, seed: image.seed }, choice);
          }}
        />
      )}
    </div>
  );
}
