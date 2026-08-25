import { useLayoutEffect, useRef, useState } from "react";

import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { aspectOf, detailOf, groupsOf, type GroupImage, type PendingSlot } from "./groups";
import { ImageTools, useImageTools, type ImageToolsState } from "./ImageTools";
import { ImageViewer } from "./ImageViewer";
import { aspectRatioOf, justifyRows } from "./justify";
import { PendingBox, PendingStatus } from "./Pending";
import type { UpscaleChoice } from "./UpscalePopover";

/** What the viewer is showing, if anything. */
type Viewing = { url: string; caption: string; detail: string };

/**
 * Horizontal space between two tiles in a row, in px.
 *
 * A number, not a CSS variable, because the row solver has to subtract it to
 * work out the scale: `width = height * sum(aspects) + gap * (n - 1)`. It is
 * `--s2` — the same 10px the rest of the studio uses — and the wall's `gap`
 * below is set from this constant so the two cannot drift apart.
 */
const GAP = 10;

/**
 * Every picture in the project and nothing else.
 *
 * The feed answers "what did I ask for, and what came back"; this answers "what
 * do I have". So there is no prompt, no model, no size and no seed on a tile —
 * not because they would not fit, but because a wall of pictures is the thing
 * the feed cannot be. What a tile *is* stays reachable: clicking one opens the
 * same viewer the feed opens, and that is where the seed and the size are.
 *
 * The image set comes from `groupsOf`, the same function the feed lays out, so
 * "every image the feed shows appears here" holds by construction rather than by
 * two flattenings agreeing. Order is the feed's too — lineages in the order they
 * were started, images within a lineage in the order they landed — because a
 * gallery that reordered would make the same project read differently in the two
 * views for no reason anybody asked for.
 *
 * Tiles load the thumbnail route, not the file: a hundred full-resolution images
 * is ~190 MB against ~2.2 MB of WebP tiles (measured). `loading="lazy"` on top
 * of that, so scrolling pays for what it reaches. The full file is fetched by
 * exactly one thing, the viewer, when a tile is actually opened.
 *
 * A tile is not always a picture. A queued or running image has a tile too — a
 * reserved box of the run's own shape, the running one carrying the same live
 * preview the feed shows — and so does a run that failed or was cancelled. This
 * view used to be given no `progress` at all, on the grounds that a gallery
 * shows what exists; the user reported the consequence, which is that a
 * submission accepted by the server left no trace here and read as a submission
 * that had failed. T8: what the server accepted is visible in every view until
 * it is done. The tile is still just a box — no prompt, no model, no seed — so
 * the rule this view was built on survives; what it lost was the claim that a
 * run in flight is somebody else's business.
 */
export function GalleryView({
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
   * The server's one global progress snapshot, shown against the running slot.
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
  const [viewing, setViewing] = useState<Viewing | null>(null);
  const tools = useImageTools();
  // Flattened out of the lineages, each image still carrying the generation that
  // owns its settings: refine and vary are the root's request, not the tile's.
  //
  // A lineage's placeholders follow its pictures, which is what "in grid order"
  // means here: the wall reads in the feed's order, and a refine in flight sits
  // with the images it was started from rather than at the end of the project.
  const tiles: Tile[] = groupsOf(generations).flatMap((group) => [
    ...group.images.map(
      (image): Tile => ({ kind: "image", key: image.url, image, root: group.root }),
    ),
    ...group.pending.map(
      (slot): Tile => ({ kind: "pending", key: `${slot.entry.id}:${slot.index}`, slot }),
    ),
  ]);

  // The one measurement this layout needs, and it is of the *container*, never
  // of an image: every tile already declares its `WxH`, so the shapes are known
  // before a byte arrives. Measuring images is what libraries like `masonic` do
  // and what produced the load jump and resize flicker this app fixed by
  // reserving ratios up front.
  const wallRef = useRef<HTMLDivElement>(null);
  // Seeded with a plausible studio width rather than 0, and this is load-bearing
  // twice over. On the first paint the ref is not attached yet, and in jsdom —
  // which has no layout engine — the measurement is 0 forever. A 0 width makes
  // the solver return no rows, so the wall would render nothing at all: no
  // pictures on the first frame in a browser, and no pictures ever under test.
  // The real width lands in the same tick via `useLayoutEffect`, before paint.
  const [width, setWidth] = useState(1000);
  useLayoutEffect(() => {
    const node = wallRef.current;
    if (node === null) return;
    // `ResizeObserver` rather than a window listener: the wall also changes width
    // when the project rail collapses, which no window event reports.
    const observer = new ResizeObserver(([entry]) => {
      // Ignore a zero: a hidden tab and jsdom both report one, and adopting it
      // would empty the wall rather than leave it as it was.
      const measured = entry?.contentRect.width ?? 0;
      if (measured > 0) setWidth(measured);
    });
    observer.observe(node);
    const initial = node.getBoundingClientRect().width;
    if (initial > 0) setWidth(initial);
    return () => observer.disconnect();
  }, []);

  // Rows aim for a sixth of the width, clamped: a target proportional to the
  // window keeps roughly the same number of pictures per row at any size, while
  // the clamp stops a phone-width window from drawing postage stamps and a
  // 2560px one from drawing banners.
  const target = Math.max(180, Math.min(340, width / 6));
  const rows = justifyRows(
    tiles.map((tile) => ({
      tile,
      aspect:
        aspectRatioOf(tile.kind === "image" ? tile.image.size : tile.slot.entry.size) ??
        16 / 9,
    })),
    { containerWidth: width, targetHeight: target, gap: GAP },
  );

  if (tiles.length === 0) {
    return (
      <div className="pg-gallery-empty">
        <p className="note">Nothing generated in this project yet.</p>
      </div>
    );
  }

  return (
    <>
      {/* Two elements, not one. `.pg-gallery` scrolls; `.pg-gallery-wall` holds
          the rows. The split dates from the multi-column layout two revisions
          back, where merging them gave the multicol box a definite height and
          made it overflow sideways — a run in flight, appended last, sat off the
          right edge. The roles stay apart for a plainer reason now: the padding
          that reserves the floating composer belongs to the thing that scrolls,
          and the wall is what gets measured. */}
      <div className="pg-gallery">
        <div className="pg-gallery-wall" ref={wallRef}>
          {rows.map((row, index) => (
            // Each row is a flex line whose items are drawn at one height and
            // their own widths. `height` is the solver's answer: the row spans
            // the wall exactly, so there is no leftover space to leave a hole
            // in — which is the whole defect this replaced.
            //
            // Keyed by the first tile in the row rather than by index alone, so
            // React does not reuse a row's DOM for a different set of pictures
            // when a new generation shifts everything along.
            <div
              className="pg-gallery-row"
              key={`${row.items[0]?.tile.key ?? "row"}:${index}`}
              style={{ height: `${row.height}px` }}
            >
              {row.items.map(({ tile, aspect }) => (
                // `flex: 0 0 auto` with an explicit width, not `flex-grow`: the
                // solver has already worked out the widths, and letting flex
                // redistribute would undo the aspect ratios it preserved.
                <div
                  className="pg-gallery-cell"
                  key={tile.key}
                  style={{ width: `${aspect * row.height}px` }}
                >
                  {tile.kind === "pending" ? (
                    <PendingTile
                      slot={tile.slot}
                      progress={progress}
                      paused={paused}
                      cancelling={cancelling}
                      onCancel={onCancel}
                    />
                  ) : (
                    <ImageTile
                      image={tile.image}
                      root={tile.root}
                      busy={busy}
                      upscalers={upscalers}
                      tools={tools}
                      nameOf={nameOf}
                      srcOf={srcOf}
                      thumbOf={thumbOf}
                      onOpen={setViewing}
                      onRefine={onRefine}
                      onVariation={onVariation}
                      onUpscale={onUpscale}
                      onDeleteImage={onDeleteImage}
                    />
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      {/* Outside the scrolling grid: the dialog belongs to the page, not to the
          wall it was opened from. */}
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

/** One cell of the wall: a picture the project has, or a run that owes one. */
type Tile =
  | { kind: "image"; key: string; image: GroupImage; root: PlaygroundGeneration }
  | { kind: "pending"; key: string; slot: PendingSlot };

function ImageTile({
  image,
  root,
  busy,
  upscalers,
  tools,
  nameOf,
  srcOf,
  thumbOf,
  onOpen,
  onRefine,
  onVariation,
  onUpscale,
  onDeleteImage,
}: {
  image: GroupImage;
  root: PlaygroundGeneration;
  busy: boolean;
  upscalers: Upscaler[];
  tools: ImageToolsState;
  nameOf: (id: string) => string;
  srcOf: (url: string) => string;
  thumbOf: (url: string) => string;
  onOpen: (viewing: Viewing) => void;
  onRefine: (root: PlaygroundGeneration, image: { url: string; seed: number }) => void;
  onVariation: (root: PlaygroundGeneration) => void;
  onUpscale: (
    root: PlaygroundGeneration,
    image: { url: string; seed: number },
    choice: UpscaleChoice,
  ) => void;
  onDeleteImage: (url: string) => void;
}) {
  return (
    <figure className="pg-tile">
      {/* A button, not a link: the click opens the viewer over the page,
          and a button is what a keyboard reaches for that. The prompt is
          the alt text and only that — an accessible name for the tile,
          not a caption drawn on it. */}
      <button
        type="button"
        className="pg-thumb"
        onClick={() =>
          onOpen({
            url: srcOf(image.url),
            caption: root.prompt,
            detail: detailOf(image, nameOf),
          })
        }
      >
        {/* The box is reserved from the run's own `WxH` before the
            thumbnail has loaded, so the column balancer has nothing left
            to re-measure: this is what stops the wall re-flowing as
            images arrive and on every frame of a window resize. An
            unparseable size leaves the ratio off and the tile sizes from
            the loaded image, exactly as it did before. */}
        <img
          src={thumbOf(image.url)}
          alt={root.prompt}
          loading="lazy"
          style={{ aspectRatio: aspectOf(image.size) ?? undefined }}
        />
      </button>
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
  );
}

/**
 * A tile for a run with no picture in it yet, or ever.
 *
 * Two things make this a tile of *this* wall rather than a feed entry dropped
 * into it. The box is reserved from the run's own `WxH` through the same
 * `aspectOf` its finished picture will use, so the columns are balanced once and
 * the wall does not move when the image lands — the flicker Step 7 fixed, which
 * a placeholder of a different shape would have reintroduced. And the words are
 * drawn *over* the box, absolutely, like the tile's toolbar: in flow they would
 * make this tile taller than the picture that replaces it, which is the same
 * re-flow by another route.
 *
 * Unlike the toolbar, they are not hidden until hover. A control you have to
 * find is bad; a *state* you have to find is the defect this whole step is
 * about.
 *
 * They also appear on one tile per *record*, not on every box it owes. An `n=2`
 * request in flight holds two boxes and has one status, and repeating the bar on
 * both read as two runs — observed in the browser, where two identical boxes are
 * visible at once and this text is the only thing that distinguishes them.
 */
function PendingTile({
  slot,
  progress,
  paused,
  cancelling,
  onCancel,
}: {
  slot: PendingSlot;
  progress: Progress;
  paused: boolean;
  cancelling: string | null;
  onCancel: (id: string) => void;
}) {
  return (
    <figure className="pg-tile pg-tile-pending" aria-label={`Generating: ${slot.prompt}`}>
      <PendingBox slot={slot} progress={progress} />
      {slot.carriesStatus && (
        <div className="pg-pending-note">
          <PendingStatus
            entry={slot.entry}
            progress={progress}
            paused={paused}
            cancelling={cancelling}
            onCancel={onCancel}
          />
        </div>
      )}
    </figure>
  );
}
