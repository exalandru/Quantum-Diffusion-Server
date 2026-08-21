import {
  Fragment,
  type CSSProperties,
  useEffect,
  useRef,
  useState,
} from "react";

import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { ImageViewer } from "./ImageViewer";
import { previewBlurPx, previewScale } from "./preview-blur";
import { Tool } from "./Tool";
import { UpscalePopover, type UpscaleChoice } from "./UpscalePopover";

/** What the viewer is showing, if anything. */
type Viewing = { url: string; caption: string; detail?: string };

/**
 * One feed entry: every generation of the same lineage, images together.
 *
 * `root` is the generation that opened the group and owns what the entry shows
 * of the request — prompt, model, size, steps, reference image. Its members keep
 * their own status, because each of them is still a separate run that can queue,
 * fail or be cancelled on its own.
 */
type Group = {
  id: string;
  root: PlaygroundGeneration;
  members: PlaygroundGeneration[];
  /**
   * Every member's images, flattened — each tagged with the member it came
   * from. The tag is needed because the entry's heading is the *root's*: an
   * upscale joins its source's lineage, so without this a 4096² image would be
   * labelled with the 512² request that produced its original.
   */
  images: GroupImage[];
  /** How many images the group's requests asked for, finished or not. */
  requested: number;
};

/** One image, plus the generation that actually produced it. */
type GroupImage = {
  url: string;
  seed: number;
  kind: PlaygroundGeneration["kind"];
  size: string;
  model: string;
};

const imagesOf = (entry: PlaygroundGeneration): GroupImage[] =>
  entry.images.map((image) => ({
    ...image,
    kind: entry.kind,
    size: entry.size,
    model: entry.model,
  }));

/** Groups in the order their roots appear: a group never moves as it grows. */
function groupsOf(generations: PlaygroundGeneration[]): Group[] {
  const groups = new Map<string, Group>();
  for (const entry of generations) {
    const group = groups.get(entry.groupId);
    if (group === undefined) {
      groups.set(entry.groupId, {
        id: entry.groupId,
        root: entry,
        members: [entry],
        images: imagesOf(entry),
        requested: entry.n,
      });
    } else {
      group.members.push(entry);
      group.images.push(...imagesOf(entry));
      group.requested += entry.n;
    }
  }
  return [...groups.values()];
}

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
  //: The image whose upscale panel is open, by URL. One at a time.
  const [upscaling, setUpscaling] = useState<string | null>(null);
  //: Remembered across images: enlarging ten of them should not cost twenty
  //: clicks re-picking the same factor and model.
  const [choice, setChoice] = useState<UpscaleChoice>({ model: "", scale: 2 });
  //: The open panel's trigger. One ref is enough because one panel is open at
  //: a time, and it is the open one whose trigger must not read as "outside".
  const upscaleTrigger = useRef<HTMLButtonElement>(null);
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
            <p className="pg-meta">
              {nameOf(root.model)} · {root.size} · {root.steps} steps · {requested}{" "}
              {requested === 1 ? "image" : "images"}
              {root.kind === "edit" && " · edit"}
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
                          // The image's own facts, not the root's: they differ
                          // for an upscale, which lives in its source's entry.
                          detail:
                            image.kind === "upscale"
                              ? `seed ${image.seed} · ${image.size} · upscaled`
                              : `seed ${image.seed} · ${image.size} · ${nameOf(image.model)}`,
                        })
                      }
                    >
                      <img src={srcOf(image.url)} alt={root.prompt} />
                    </button>
                    {image.kind === "upscale" && (
                      <figcaption className="pg-image-tag">Upscaled · {image.size}</figcaption>
                    )}
                    <div className="pg-image-actions">
                    <div className="pg-image-tools" role="toolbar" aria-label="Image actions">
                      <Tool
                        tip="Refine"
                        disabled={busy}
                        // The bare image, not the group's annotated copy: `kind`,
                        // `size` and `model` are this component's bookkeeping for
                        // the badge, and leaking them would make them contract.
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
                        expanded={upscaling === image.url}
                        buttonRef={upscaling === image.url ? upscaleTrigger : undefined}
                        onClick={() =>
                          setUpscaling((open) => {
                            if (open === image.url) return null;
                            // Default to the first catalogue entry rather than
                            // a hardcoded id: the catalogue is the server's.
                            if (!upscalers.some((entry) => entry.id === choice.model)) {
                              setChoice((current) => ({
                                ...current,
                                model: upscalers[0]?.id ?? "",
                              }));
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
                          // The convention for a destructive one-click action
                          // here — SessionList does the same for whole sessions.
                          if (window.confirm("Delete this image?")) onDeleteImage(image.url);
                        }}
                      >
                        <path d="M4 7h16M9 7V5h6v2M6.5 7l.8 12h9.4l.8-12M10 11v5M14 11v5" />
                      </Tool>
                    </div>
                    {upscaling === image.url && (
                      <UpscalePopover
                        upscalers={upscalers}
                        choice={choice}
                        onChoose={setChoice}
                        trigger={upscaleTrigger}
                        onClose={() => setUpscaling(null)}
                        onSubmit={() => {
                          setUpscaling(null);
                          onUpscale(root, { url: image.url, seed: image.seed }, choice);
                        }}
                      />
                    )}
                    </div>
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
              <Fragment key={member.id}>
                {member.status === "queued" && (
                  <div className="pg-status">
                    <span className="note">{paused ? "Held — the queue is paused." : "Queued…"}</span>
                    <button
                      className="small"
                      disabled={cancelling === member.id}
                      onClick={() => onCancel(member.id)}
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {member.status === "running" && (
                  <Running
                    entry={member}
                    progress={progress}
                    onCancel={onCancel}
                    cancelling={cancelling === member.id}
                  />
                )}

                {member.status === "failed" && (
                  <div className="notice notice-error" role="status">
                    <strong>Generation failed.</strong> {member.error}
                  </div>
                )}

                {member.status === "cancelled" && <p className="note">Cancelled.</p>}
              </Fragment>
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

function Running({
  entry,
  progress,
  onCancel,
  cancelling,
}: {
  entry: PlaygroundGeneration;
  progress: Progress;
  onCancel: (id: string) => void;
  cancelling: boolean;
}) {
  const upscaling = entry.kind === "upscale";
  const stepping =
    (progress.state === "generating" || progress.state === "upscaling") && progress.total > 0;
  const percent = stepping ? Math.min(100, (progress.step / progress.total) * 100) : 0;
  return (
    <div className="pg-running">
      {/* The same two shapes the dashboard uses: a measured fill when there are
          steps to count, and an indeterminate track while the weights load or
          while the engine is busy with somebody else's request. A bar with an
          invented denominator says less than a moving one. */}
      {stepping ? (
        <div className="bar">
          <div className="bar-fill" style={{ width: `${percent}%` }} />
        </div>
      ) : (
        <div className="bar bar-indeterminate" />
      )}
      <div className="pg-status">
        <span className="note">
          {stepping
            ? upscaling
              ? `Upscaling - tile ${progress.step} of ${progress.total}`
              : `Image ${Math.min(entry.images.length + 1, entry.n)} of ${entry.n} - step ${progress.step} of ${progress.total}`
            : upscaling
              ? "Loading the upscaler…"
              : "Loading the model…"}
          {progress.elapsed_s !== null && stepping && ` · ${progress.elapsed_s.toFixed(0)}s`}
        </span>
        <button className="small" disabled={cancelling} onClick={() => onCancel(entry.id)}>
          {cancelling ? "Cancelling…" : "Cancel"}
        </button>
      </div>
    </div>
  );
}

/** Must match the `.pg-preview img` opacity transition in `styles.css`. */
const FADE_MS = 500;

/**
 * The partially-denoised image, refreshed as the server decodes new frames.
 *
 * A frame is fetched — not streamed — the moment `seq` changes: the bytes live
 * in one server-side slot at `/playground/api/preview`, and `seq` is both the
 * change signal and the cache-buster. Two frames are on screen at once at most:
 * the new one fades in over the old, which is dropped once the fade is over.
 *
 * The frames are blurred in proportion to how far the run has got — see
 * `preview-blur.ts` for why, and why the amount depends on the step count as
 * well as the progress. The value is set as a custom property on the box rather
 * than on each frame, so both sides of a crossfade always agree on it.
 */
function StepPreview({
  seq,
  size,
  step,
  total,
}: {
  seq: number;
  size: string;
  step: number;
  total: number;
}) {
  const [frames, setFrames] = useState<{ seq: number; on: boolean }[]>([]);
  const sweep = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // 0 means "no frame for what is running now": a fresh image of an n>1
    // request, or somebody else's `/v1` job holding the engine. Back to the
    // placeholder rather than showing a stale partial image.
    if (seq === 0) {
      setFrames([]);
      return;
    }
    setFrames((current) => [...current, { seq, on: false }].slice(-2));
  }, [seq]);

  useEffect(() => () => clearTimeout(sweep.current ?? undefined), []);

  // `WxH` as the request recorded it; anything else falls back to a square box
  // rather than collapsing the placeholder to nothing.
  const [w = 0, h = 0] = size.split("x").map(Number);
  const square = !(Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0);
  const ratio = square ? "1 / 1" : `${w} / ${h}`;

  const blur = previewBlurPx(step, total);

  return (
    <div
      className="pg-preview"
      style={
        {
          aspectRatio: ratio,
          "--pg-preview-blur": `${blur.toFixed(2)}px`,
          "--pg-preview-scale": previewScale(blur).toFixed(4),
        } as CSSProperties
      }
    >
      {frames.map((frame) => (
        <img
          key={frame.seq}
          className={frame.on ? "on" : undefined}
          src={`/playground/api/preview?v=${frame.seq}`}
          alt=""
          aria-hidden="true"
          onLoad={() => {
            setFrames((current) =>
              current.map((f) => (f.seq === frame.seq ? { ...f, on: true } : f)),
            );
            clearTimeout(sweep.current ?? undefined);
            // Once this frame has finished fading in, everything under it is
            // invisible and can go.
            sweep.current = setTimeout(() => {
              setFrames((current) => current.filter((f) => f.seq >= frame.seq));
            }, FADE_MS);
          }}
          // The run ended between the SSE frame and this fetch, so the slot is
          // already empty: drop this frame, keep whatever is showing.
          onError={() => setFrames((current) => current.filter((f) => f.seq !== frame.seq))}
        />
      ))}
      {/* Above the frames, and there even when none has arrived: a slow glow
          crossing the box is what says "still being made" about an image that
          will not change for the next few seconds. */}
      <div className="pg-preview-glow" aria-hidden="true" />
    </div>
  );
}
