import { type ReactNode, useState } from "react";

import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { detailOf, groupsOf, type GroupImage, type PendingSlot } from "./groups";
import { ImageTools, useImageTools } from "./ImageTools";
import { ImageViewer } from "./ImageViewer";
import { PendingBox, PendingStatus } from "./Pending";
import type { UpscaleChoice } from "./UpscalePopover";

/**
 * One frame of the strip: a picture with the request that owns its prompt, or a
 * run that owes a picture.
 *
 * `key` is what the held selection is compared against, and for a picture it is
 * the image's URL — the value this view held before runs in flight were frames
 * at all, so a project with nothing in flight resolves exactly as it did.
 */
type Frame =
  | { kind: "image"; key: string; image: GroupImage; root: PlaygroundGeneration }
  | { kind: "pending"; key: string; slot: PendingSlot };

/**
 * One picture at a time, with everything known about it beside it.
 *
 * The third presentation of the same project. The feed answers "what did I ask
 * for, and what came back"; the gallery answers "what do I have"; this answers
 * "what is *this* one" — a single image on the stage, the rest of the project as
 * a strip under it, and the facts of the selected one in a panel that does not
 * have to be opened. It is the only view where metadata is on screen without a
 * click, which is why it is also the only one that has to be careful about
 * *whose* metadata that is (see `factsOf` below).
 *
 * The image set and its order are `groupsOf`'s, exactly as the gallery's are:
 * every image the feed shows appears here, lineages in the order they were
 * started, images within a lineage in the order they landed. Selecting is the
 * only thing this view adds to the project, and it adds nothing to the record.
 *
 * The strip loads thumbnails (~22 KB against ~1.9 MB, measured) with
 * `loading="lazy"`; the stage loads the one full-resolution file the user is
 * actually looking at. That split is the whole reason a filmstrip of a
 * hundred-image project is affordable.
 *
 * A frame is not always a picture. A queued or running image is a frame of the
 * strip too, and the stage shows it when it is the newest thing in the project:
 * this view used to be given no `progress` on the grounds that it shows a
 * picture that exists, and the user reported the consequence — a submission the
 * server had accepted left the hero sitting on an old picture with nothing to
 * say why. T8: what the server accepted is visible in every view until it is
 * done. A run that failed or was cancelled keeps a frame as well, for the same
 * reason in its worst case: a run that ends badly and vanishes is the same
 * defect wearing a different hat.
 */
export function LightTableView({
  generations,
  progress,
  paused,
  cancelling,
  onCancel,
  busy,
  upscalers,
  nameOf,
  srcOf,
  thumbOf,
  onRefine,
  onVariation,
  onUpscale,
  onDeleteImage,
}: {
  generations: PlaygroundGeneration[];
  /**
   * The server's one global progress snapshot, shown against the running frame.
   *
   * The same single snapshot the feed reads, with the same limitation: an
   * external `/v1` client generating at the same time is the one case where the
   * step counter belongs to someone else, and it is not worth a second
   * mechanism to tell the two apart.
   */
  progress: Progress;
  /** The queue is held: what is queued says so instead of pretending to wait its turn. */
  paused: boolean;
  /** The generation whose cancel is in flight, or null. */
  cancelling: string | null;
  onCancel: (id: string) => void;
  /** A submission is in flight: the generating actions wait for it. */
  busy: boolean;
  upscalers: Upscaler[];
  /** Id → readable name; the record stores the API id. */
  nameOf: (id: string) => string;
  /** Server URL → loadable full-resolution `src`, token included. */
  srcOf: (url: string) => string;
  /** Server URL → loadable thumbnail `src`, token included. */
  thumbOf: (url: string) => string;
  onRefine: (root: PlaygroundGeneration, image: { url: string; seed: number }) => void;
  onVariation: (root: PlaygroundGeneration) => void;
  onUpscale: (
    root: PlaygroundGeneration,
    image: { url: string; seed: number },
    choice: UpscaleChoice,
  ) => void;
  onDeleteImage: (url: string) => void;
}) {
  /**
   * Which frame is on the stage — held as the frame's key, not as an index.
   *
   * View-local on purpose: it is not a fact about the project, so it has no
   * business in `PlaygroundApp`'s state, which is the server's view of the
   * world. Nothing persists it either, so a reload opens on the first image,
   * and `PlaygroundApp` keys this component on the selected project so a
   * project switch takes the selection with it (see the call site).
   *
   * A key rather than an index, resolved by lookup below, so that the ways the
   * project's frames change *under* a mounted view resolve to something
   * showable without an effect to keep the two in step: deleting the selected
   * image, and a refine or an upscale landing new ones, both fall back to the
   * default frame rather than leaving an index pointing at a different picture —
   * or past the end of the strip. For a picture the key *is* its URL, which is
   * what it was before runs in flight had frames of their own.
   */
  const [held, setHeld] = useState<string | null>(null);
  /** Whether the stage's picture is also open in the viewer, over the page. */
  const [zoomed, setZoomed] = useState(false);
  const tools = useImageTools();

  // Pictures and runs-without-a-picture in one strip, a lineage's slots after
  // its images — the gallery's ordering rule and the feed's, so the three views
  // read the same project the same way round.
  //
  // One frame per pending *record*, not per box it owes, which is where this
  // parts company with the wall. The gallery reserves the place each picture will
  // take, so an `n=2` request holds two boxes there; a filmstrip is a row of
  // things to look at, and two identical empty boxes for one request is padding
  // that also puts the stage's default on a box with nothing to say. The slot
  // that carries the record's status is the record's one frame.
  const frames: Frame[] = groupsOf(generations).flatMap((group) => [
    ...group.images.map(
      (image): Frame => ({ kind: "image", key: image.url, image, root: group.root }),
    ),
    ...group.pending
      .filter((slot) => slot.carriesStatus)
      .map((slot): Frame => ({ kind: "pending", key: `pending:${slot.entry.id}:${slot.index}`, slot })),
  ]);
  const found = frames.findIndex((frame) => frame.key === held);
  // Nothing held, or holding a frame this project no longer has.
  //
  // A run genuinely *in flight* wins: it is the last frame of the last lineage,
  // and "the hero must not sit on an old picture with no explanation while
  // something is being made" is the whole of that exception.
  //
  // `kind === "pending"` alone is not that test, and reading it as one is what
  // put "Generation failed. Interrupted by server restart" on the stage of a
  // project holding twelve finished pictures. A frame is `pending` whenever its
  // record owes no file — queued, running, failed *and* cancelled all qualify —
  // so a project whose last run failed took the in-flight branch and pinned the
  // stage to the failure. Measured in the browser: sixteen tiles, `aria-current`
  // on the sixteenth, which was the marker rather than a picture.
  //
  // Only queued and running are worth pre-empting a picture for. A failure has
  // already happened; it keeps its place in the strip and says so there.
  const last = frames.at(-1);
  const inFlight =
    last?.kind === "pending" &&
    (last.slot.entry.status === "queued" || last.slot.entry.status === "running");
  // Otherwise the first *picture*, not the first frame — those differ exactly
  // when a project opens on a lineage that produced nothing.
  //
  // A project with nothing but failures falls back to the first frame, which is
  // the failure: there is nothing better to show, and showing it is how the view
  // says so.
  const firstPicture = frames.findIndex((frame) => frame.kind === "image");
  const fallback = inFlight ? frames.length - 1 : firstPicture === -1 ? 0 : firstPicture;
  const at = found === -1 ? fallback : found;
  const current = frames[at];

  if (current === undefined) {
    // Nothing at all: no images, and nothing queued, running, failed or
    // cancelled either. Same empty state as the gallery's, and the same class —
    // the question ("a view with nothing to draw") and the answer (say so,
    // centred) are identical.
    return (
      <div className="pg-gallery-empty">
        <p className="note">Nothing generated in this project yet.</p>
      </div>
    );
  }

  const previous = frames[at - 1];
  const next = frames[at + 1];

  return (
    <>
      <div className="pg-table">
        <div className="pg-table-stage">
          <div className="pg-table-main">
            {/* Both affordances are rendered even at the ends of the strip, and
                disabled there: a control that disappears at the first image
                moves the one beside it, and the arrows are what the eye returns
                to between pictures. */}
            <button
              type="button"
              className="pg-table-nav prev"
              aria-label="Previous image"
              disabled={previous === undefined}
              onClick={() => previous && setHeld(previous.key)}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="m15 6-6 6 6 6" />
              </svg>
            </button>
            {current.kind === "image" ? (
              /* A button, not a link: the click opens the viewer over the page,
                 and a button is what a keyboard reaches for that. The full file
                 rather than the tile — this is the one picture being looked at,
                 and the strip below is what pays for the rest in thumbnails. */
              <button type="button" className="pg-table-hero" onClick={() => setZoomed(true)}>
                <img src={srcOf(current.image.url)} alt={current.root.prompt} />
              </button>
            ) : (
              /* The stage, given to a run that has no file yet: the same
                 reserved box and the same live preview the gallery's placeholder
                 and the feed's cell use, with the run's state and its cancel
                 under it. Not a button — there is nothing to zoom into — and
                 nothing here fetches an image route. */
              <div className="pg-table-pending">
                <PendingBox slot={current.slot} progress={progress} />
                <PendingStatus
                  entry={current.slot.entry}
                  progress={progress}
                  paused={paused}
                  cancelling={cancelling}
                  onCancel={onCancel}
                />
              </div>
            )}
            <button
              type="button"
              className="pg-table-nav next"
              aria-label="Next image"
              disabled={next === undefined}
              onClick={() => next && setHeld(next.key)}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="m9 6 6 6-6 6" />
              </svg>
            </button>
          </div>
          {/* The whole project, one row, scrolling sideways. `aria-current` and
              not `aria-selected`: these are not tabs, and "the one you are
              looking at" is exactly what `current` means. */}
          <div className="pg-strip" role="list" aria-label="Project images">
            {frames.map((frame) => (
              <button
                type="button"
                role="listitem"
                key={frame.key}
                className={
                  frame.kind === "pending" ? "pg-strip-tile pg-strip-pending" : "pg-strip-tile"
                }
                aria-current={frame.key === current.key ? "true" : undefined}
                {...(frame.kind === "pending"
                  ? {
                      // What this frame *is*, which is not always "generating".
                      // A pending frame covers queued, running, failed and
                      // cancelled alike, so the old unconditional "Generating:"
                      // announced a run that had already failed as though it
                      // were still working — and, with the stage now defaulting
                      // to a picture, the strip is the only place a failure is
                      // announced at all.
                      //
                      // `title` as well as `aria-label`: the error is the reason
                      // to click this tile, and hovering it should say so rather
                      // than requiring the click first.
                      "aria-label": labelOfPending(frame.slot),
                      title: frame.slot.entry.error ?? undefined,
                    }
                  : {})}
                onClick={() => setHeld(frame.key)}
              >
                {frame.kind === "pending" ? (
                  // Selectable like any other frame: a user who left the run to
                  // look at an old picture has to be able to go back to watching
                  // it, and the strip is where the frames are.
                  <PendingBox slot={frame.slot} progress={progress} />
                ) : (
                  <img src={thumbOf(frame.image.url)} alt={frame.root.prompt} loading="lazy" />
                )}
              </button>
            ))}
          </div>
        </div>
        {current.kind === "image" ? (
          <Inspector facts={current.image} root={current.root} seed={current.image.seed} nameOf={nameOf}>
            <section className="pg-insp-section">
              <h3>Actions</h3>
              {/* The same four actions as the other two views, from the same
                  component: refine and vary read the root's settings, upscale
                  names a file the server already owns, delete costs a picture.
                  Here they are always visible — there is one image on the stage,
                  so there is nothing for a hover to disambiguate. */}
              <ImageTools
                image={current.image}
                root={current.root}
                busy={busy}
                upscalers={upscalers}
                tools={tools}
                onRefine={onRefine}
                onVariation={onVariation}
                onUpscale={onUpscale}
                onDeleteImage={onDeleteImage}
              />
            </section>
          </Inspector>
        ) : (
          /* The same panel for a run with no file: its settings are facts about
             the request, which the record carries in full, and the seed is the
             one thing it cannot state yet — a seed is assigned per image, and
             `seeds` is a list the images are matched to as they land. So the row
             is dropped rather than filled with the first of them, which would be
             a claim about a picture nobody has seen.

             No section of its own for the run's state, and no second cancel: both
             are on the stage, an inch to the left, where the box they are about
             is. Saying it twice on one screen is not twice as clear, and a page
             with two Cancel buttons for one run invites the question of whether
             they do the same thing. */
          <Inspector facts={current.slot.entry} root={current.slot.entry} seed={null} nameOf={nameOf} />
        )}
      </div>
      {/* Outside the frame: the dialog belongs to the page, not to the stage it
          was opened from. It follows the stage while it is open, because the
          arrows keep working underneath it. */}
      {zoomed && current.kind === "image" && (
        <ImageViewer
          url={srcOf(current.image.url)}
          caption={current.root.prompt}
          detail={detailOf(current.image, nameOf)}
          onClose={() => setZoomed(false)}
        />
      )}
    </>
  );
}

/** What a strip tile announces, which is not always "generating". */
function labelOfPending(slot: PendingSlot): string {
  const status = slot.entry.status;
  if (status === "failed") {
    // The reason, not just the fact: with the stage defaulting to a picture,
    // this tile is where a failure is stated, and "it failed" without "why" is
    // a dead end.
    return slot.entry.error ? `Generation failed: ${slot.entry.error}` : "Generation failed";
  }
  if (status === "cancelled") return "Generation cancelled";
  if (status === "queued") return `Queued: ${slot.prompt}`;
  return `Generating: ${slot.prompt}`;
}

/**
 * The selected frame's facts.
 *
 * Which field comes from where is the correctness of this whole view, so it is
 * spelled out rather than left to the shape of the data:
 *
 * - **Prompt and enhanced prompt come from the group's root.** An upscale has no
 *   prompt of its own — the server copies its source's into the record for the
 *   feed's benefit — and neither does a refine's parent request describe the
 *   refine. The prompt of a lineage is the lineage's, and the root owns it.
 * - **Model, size, steps and seed come from the image itself.** These are facts
 *   about the run that produced this file, and for an upscale they are *not* the
 *   root's: an upscale joins its source's lineage, so a 4096² image sits in a
 *   group whose root asked for 512² with a diffusion model. Reading the root
 *   here would label that picture 512², which is precisely the mislabelling
 *   `GroupImage` was introduced to make impossible. `groups.ts` tags every
 *   flattened image with its own generation's `kind`, `size`, `model` and
 *   `steps`; this panel spends all four.
 *
 * `facts` is that image — or, for a run with no picture yet, the record of the
 * run itself, which carries the same four fields and is the honest answer to
 * "what is being made". Hence a `Facts` of exactly those four rather than a
 * `GroupImage`: the panel never had any use for the URL, and asking for less is
 * what lets one panel serve both frames instead of two panels drifting apart.
 * `seed` is passed separately and may be null, because a seed belongs to an
 * image and a queued request has a *list* of them that its images are matched
 * to as they land.
 *
 * The last section is the caller's: the actions of a picture and the state of a
 * run are not the same question, and a `kind`-shaped conditional in here would
 * be this component knowing about both.
 */
function Inspector({
  facts,
  root,
  seed,
  nameOf,
  children,
}: {
  facts: Facts;
  root: PlaygroundGeneration;
  seed: number | null;
  nameOf: (id: string) => string;
  children?: ReactNode;
}) {
  return (
    <aside className="pg-inspector" aria-label="Image details">
      <section className="pg-insp-section">
        <h3>Prompt</h3>
        <p className="pg-insp-prompt">{root.prompt}</p>
      </section>
      {/* Only when there is one, and never as the title: the same rule the feed
          follows — nobody is shown words they did not write as though they had
          written them.

          Collapsed, and by the feed's own mechanism. An enhanced prompt runs to
          a couple of hundred words; expanded by default it filled the whole
          panel, clipped mid-sentence, and pushed Settings and Actions below the
          fold — so compacting the settings bought height that this then spent.
          The feed had already answered this with a `<details>`; a second
          treatment of the same text in the same app was the anomaly. */}
      {root.rewrittenPrompt && (
        <section className="pg-insp-section">
          <details className="pg-rewrite">
            <summary>Enhanced prompt</summary>
            <p className="pg-rewrite-text pg-insp-enhanced">{root.rewrittenPrompt}</p>
          </details>
        </section>
      )}
      <section className="pg-insp-section">
        <h3>
          Settings
          {/* What explains the missing step count below, rather than a "0" that
              reads as a run that never sampled. */}
          {facts.kind === "upscale" && <span className="pg-insp-tag">Upscaled</span>}
        </h3>
        {/* Chips on one wrapping line, not a stacked definition list. The list
            spent a full row per fact — four labels, four values, four rows —
            for four short values that read perfectly well as chips. That height
            came out of the prompt above it, which is the one thing here that
            genuinely needs the room. The same facts, in the same chips the feed
            and the gallery already use.

            The `dl` stays: these *are* term/value pairs, and flattening them to
            plain text would say less. Only the layout changed — `dt` is now the
            chip's own leading word rather than a line of its own. */}
        <dl className="pg-insp-facts pg-insp-facts-inline">
          <div className="pg-kv">
            <dt>Model</dt>
            {/* `title` because the chip ellipsises: "stabilityai/stable-diffusion-3.5-large"
                does not fit a 280px panel on one line, and the full name has to
                stay reachable without opening anything. */}
            <dd className="pill pill-fact" title={nameOf(facts.model)}>
              {nameOf(facts.model)}
            </dd>
          </div>
          <div className="pg-kv">
            <dt>Size</dt>
            <dd className="pill pill-fact">{facts.size}</dd>
          </div>
          {/* An upscale's record carries `steps=0` because a super-resolution
              pass has no denoising loop. Dropping the row states less than a
              zero would, and a zero would state something false. */}
          {facts.steps > 0 && (
            <div className="pg-kv">
              <dt>Steps</dt>
              <dd className="pill pill-fact">{facts.steps}</dd>
            </div>
          )}
          {seed !== null && (
            <div className="pg-kv">
              <dt>Seed</dt>
              <dd className="pill pill-fact">{seed}</dd>
            </div>
          )}
        </dl>
      </section>
      {children}
    </aside>
  );
}

/**
 * What the panel states about whatever is on the stage.
 *
 * Satisfied by a `GroupImage` and by a `PlaygroundGeneration` alike, which is
 * the point: the four settings are properties of a *run*, and an image carries
 * them because `groups.ts` tags it with its own run's.
 */
type Facts = {
  kind: PlaygroundGeneration["kind"];
  size: string;
  model: string;
  steps: number;
};
