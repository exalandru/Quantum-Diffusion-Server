import { useEffect, useRef, useState } from "react";

import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { detailOf, groupsOf } from "./groups";
import { ImageTools, useImageTools } from "./ImageTools";
import { ImageViewer } from "./ImageViewer";
import { PendingStatus, StepPreview } from "./Pending";
import type { UpscaleChoice } from "./UpscalePopover";

/** What the viewer is showing, if anything. */
type Viewing = { url: string; caption: string; detail?: string };

/**
 * The transcript of a session: every generation, oldest first, with its images.
 *
 * Generations of one lineage are drawn as a single entry, so refining or varying
 * an image adds to the picture it came from instead of restating the prompt
 * below it — the same shape a single request for several images produces.
 *
 * Progress is the server's one global snapshot, shown against the generation
 * that is `running`. With a single-generation engine those are the same thing;
 * an external `/v1` client generating at the same time is the one case where the
 * step counter belongs to someone else, and it is not worth a second mechanism.
 */
export function GenerationFeed({
  generations,
  progress,
  onCancel,
  cancelling,
  busy,
  onRefine,
  onVariation,
  onUpscale,
  upscalers,
  onDeleteImage,
  onDeleteGroup,
  onUsePrompt,
  paused,
  nameOf,
  srcOf = (url) => url,
}: {
  generations: PlaygroundGeneration[];
  progress: Progress;
  onCancel: (id: string) => void;
  cancelling: string | null;
  /** A submission is in flight: the generating actions wait for it. */
  busy: boolean;
  /** The clicked image, and the root whose settings a new image reuses. */
  onRefine: (root: PlaygroundGeneration, image: { url: string; seed: number }) => void;
  onVariation: (root: PlaygroundGeneration) => void;
  /** Put an enhanced prompt back in the composer, as an ordinary prompt. */
  onUsePrompt?: (prompt: string) => void;
  /** The clicked image, and the factor and model chosen for it. */
  onUpscale: (
    root: PlaygroundGeneration,
    image: { url: string; seed: number },
    choice: UpscaleChoice,
  ) => void;
  /**
   * The upscaler catalogue, or an empty list.
   *
   * Empty means the control stays disabled — the same fail-closed rule the
   * composer's drop zone and the advanced fields follow: an offer the server
   * may refuse is worse than no offer.
   */
  upscalers: Upscaler[];
  onDeleteImage: (url: string) => void;
  /** The whole entry: every generation of the lineage, and the files only it owned. */
  onDeleteGroup: (groupId: string) => void;
  /** The queue is held: what is queued says so instead of pretending to wait its turn. */
  paused: boolean;
  /** Id → readable name; the record stores the API id. */
  nameOf: (id: string) => string;
  /**
   * Server URL → loadable `src`. A locked session's images need its unlock
   * token on the URL; the URLs in the records stay the server's.
   */
  srcOf?: (url: string) => string;
}) {
  const end = useRef<HTMLDivElement>(null);
  const [viewing, setViewing] = useState<Viewing | null>(null);
  const tools = useImageTools();
  const groups = groupsOf(generations);
  // An entry that *appears* scrolls the feed down, and nothing else does. A new
  // prompt is written in the composer at the bottom of the page, so that is
  // where the eye already is; everything else happens where the user is already
  // looking. Refining or varying grows an entry they may have scrolled up to on
  // purpose — neither the click nor the image landing later moves the feed — and
  // deleting an image, which can dissolve a whole entry, must not either.
  //
  // Hence "an id that was not there before", not "the set of ids changed": the
  // latter also fires when an entry disappears. Switching session is the one
  // other case that scrolls, because the newest entry is the one to land on.
  const seen = useRef<{ session: string | null; ids: Set<string> }>({
    session: generations[0]?.sessionId ?? null,
    ids: new Set(),
  });
  const session = generations[0]?.sessionId ?? null;
  const ids = groups.map((group) => group.id);
  const appeared = ids.some((id) => !seen.current.ids.has(id));
  const switched = seen.current.session !== session;
  useEffect(() => {
    seen.current = { session, ids: new Set(ids) };
    if (appeared || switched) end.current?.scrollIntoView({ block: "end" });
  });

  return (
    <>
      <div className="pg-feed">
        {groups.map(({ id, root, members, images, requested }) => {
          // At most one member of one group can be running: the runner is a
          // single FIFO worker and the engine takes one job at a time.
          const running = members.find((member) => member.status === "running");
          return (
          <article className="pg-entry" key={id}>
            <div className="pg-entry-head">
              <p className="pg-prompt">{root.prompt}</p>
              {/* Revealed on hover or keyboard focus, like the session row's
                  actions: a control a keyboard user cannot reach is not a
                  control. `window.confirm` is the convention here for a
                  destructive one click — the session list and the per-image
                  tool both do it, and a misfire costs the whole entry. */}
              <div className="pg-entry-actions">
                <button
                  type="button"
                  className="small danger"
                  aria-label={`Delete entry: ${root.prompt}`}
                  onClick={() => {
                    const count = images.length;
                    const what =
                      count === 0
                        ? "Delete this entry?"
                        : `Delete this entry and its ${count} ${count === 1 ? "image" : "images"}?`;
                    if (window.confirm(what)) onDeleteGroup(id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
            {root.negativePrompt && (
              <p className="pg-prompt-negative">
                <span className="pg-negative-tag">Negative</span> {root.negativePrompt}
              </p>
            )}
            {/*
              Behind a disclosure, never as the title. The entry is titled with
              what the user typed, so nobody is ever shown words they did not
              write as though they had written them. Opening it shows the text
              the image was actually made from -- and "Use this prompt" puts it
              in the composer as an ordinary prompt, which is the whole of
              "edit" and "pin" without a second mechanism: a pinned rewrite is
              just a typed prompt.
            */}
            {root.rewrittenPrompt && (
              <details className="pg-rewrite">
                <summary>Enhanced prompt</summary>
                <p className="pg-rewrite-text">{root.rewrittenPrompt}</p>
                {onUsePrompt && (
                  <button
                    className="small"
                    onClick={() => onUsePrompt(root.rewrittenPrompt as string)}
                  >
                    Use this prompt
                  </button>
                )}
              </details>
            )}
            {root.rewriteError && (
              <p className="pg-rewrite-failed">
                Enhancing failed ({root.rewriteError}) — generated from your prompt.
              </p>
            )}
            {/* One chip per fact, not a sentence of them separated by dots. The
                facts are independent — model, size, step count, how many images
                were asked for — and a dotted run of four reads as one clause you
                have to parse. `.pill-fact` is the sheet's pill in its quiet
                voice; the loud variants report a *state*, and a run's settings
                are not a state. */}
            <p className="pg-meta">
              <span className="pill pill-fact">{nameOf(root.model)}</span>
              <span className="pill pill-fact">{root.size}</span>
              <span className="pill pill-fact">{root.steps} steps</span>
              <span className="pill pill-fact">
                {requested} {requested === 1 ? "image" : "images"}
              </span>
              {root.kind === "edit" && <span className="pill pill-fact">edit</span>}
            </p>

            {root.contextImage && (
              <button
                type="button"
                className="pg-thumb pg-context"
                onClick={() =>
                  setViewing({ url: srcOf(root.contextImage as string), caption: "Reference image" })
                }
              >
                <img src={srcOf(root.contextImage)} alt="Reference image" />
              </button>
            )}

            {/* The grid also exists for a run with no image yet: the preview
                takes the cell the image will land in, so a refine or a variation
                grows the entry in place instead of pushing a box underneath it. */}
            {(images.length > 0 || running !== undefined) && (
              <div className="pg-images">
                {images.map((image) => (
                  // `position: relative` comes from the sheet: the tag and the
                  // toolbar are drawn *on* the picture now, the treatment the
                  // gallery's tiles already use. In flow, a cell with a
                  // "Upscaled · 2880x1600" caption was taller than its
                  // neighbours, so its toolbar sat lower than theirs and a row
                  // of four pictures looked broken. Nothing below the picture
                  // means every cell of a row is the height of its picture.
                  <figure className="pg-image-cell" key={image.url}>
                    {/* A button, not a link: the click opens the viewer over the
                        page, and a button is what a keyboard reaches for that. */}
                    <button
                      type="button"
                      className="pg-thumb"
                      onClick={() =>
                        setViewing({
                          url: srcOf(image.url),
                          caption: root.prompt,
                          detail: detailOf(image, nameOf),
                        })
                      }
                    >
                      <img src={srcOf(image.url)} alt={root.prompt} />
                    </button>
                    {image.kind === "upscale" && (
                      <figcaption className="pg-image-tag">Upscaled · {image.size}</figcaption>
                    )}
                    <ImageTools
                      image={image}
                      root={root}
                      busy={busy}
                      upscalers={upscalers}
                      tools={tools}
                      onRefine={onRefine}
                      onVariation={onVariation}
                      onUpscale={onUpscale}
                      onDeleteImage={onDeleteImage}
                    />
                  </figure>
                ))}
                {running !== undefined && running.kind !== "upscale" && (
                  <StepPreview
                    seq={progress.state === "generating" ? progress.preview_seq : 0}
                    size={running.size}
                    // Guarded like `seq`, and for the same reason: while the
                    // weights load, or while an external `/v1` client holds the
                    // engine, that step count is not this preview's.
                    step={progress.state === "generating" ? progress.step : 0}
                    total={progress.state === "generating" ? progress.total : 0}
                  />
                )}
              </div>
            )}

            {/* Per member, not per entry: one image of the group can still be
                queued while another has already failed. */}
            {members.map((member) => (
              <PendingStatus
                key={member.id}
                entry={member}
                progress={progress}
                paused={paused}
                cancelling={cancelling}
                onCancel={onCancel}
              />
            ))}
          </article>
          );
        })}
        <div ref={end} />
      </div>
      {/* Outside the scrolling feed: the dialog belongs to the page, not to the
          transcript it was opened from. */}
      {viewing && (
        <ImageViewer
          url={viewing.url}
          caption={viewing.caption}
          detail={viewing.detail}
          onClose={() => setViewing(null)}
        />
      )}
    </>
  );
}
