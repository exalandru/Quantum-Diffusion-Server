/**
 * What a run with no picture looks like, wherever it is drawn.
 *
 * Extracted from `GenerationFeed` when the gallery and the light table were made
 * to show work in flight (T8: in every view, a submission the server accepted is
 * visibly in flight until it completes, fails or is cancelled). The feed had all
 * of it — the denoising preview, the measured bar, the "Queued…" line, the
 * failure notice and the cancel button — and the alternative to moving it here
 * was three views each writing their own words for the same four states. The
 * precedent is `groups.ts` and `ImageTools.tsx`: the property that has to hold is
 * that the views *agree*, and agreement is a consequence of one implementation,
 * not of three that currently match.
 *
 * What is deliberately *not* here is where any of it goes. A wall of tiles, a
 * filmstrip and a transcript place a placeholder differently, and bolting the
 * feed's entry onto the other two would make them the feed. Each view owns its
 * own geometry and spends these three pieces inside it:
 *
 * - `PendingBox` — the reserved box a picture will land in, showing the live
 *   preview when this is the slot the engine is working on;
 * - `PendingStatus` — the words for the run's state, and the cancel button;
 * - `StepPreview` — the frames themselves, used through `PendingBox`.
 */
import { type CSSProperties, useEffect, useRef, useState } from "react";

import type { PlaygroundGeneration, Progress } from "../types";
import { aspectOf, type PendingSlot } from "./groups";
import { previewBlurPx, previewScale } from "./preview-blur";

/**
 * The box a slot's picture will take, before there is a picture.
 *
 * The aspect ratio comes from `aspectOf` — the same function the gallery's tiles
 * reserve their boxes with — so a placeholder and the finished image it gives way
 * to are the same shape, and the wall does not re-flow when one replaces the
 * other. That is the Step 7 flicker fix, and a placeholder that ignored it would
 * reintroduce exactly what it fixed. A square is the fallback when the run's
 * `WxH` does not parse, which no server record does: `height: auto` on a box with
 * no picture in it is a box with no height.
 *
 * The live slot renders `StepPreview` whether or not the engine is generating
 * yet, with the step counters guarded the way the feed guards them: while the
 * weights load, or while an external `/v1` client holds the engine, that step
 * count is not this preview's. Keeping the component mounted across that
 * transition is what stops the frames it has collected being thrown away when
 * the state flickers.
 */
export function PendingBox({
  slot,
  progress,
}: {
  slot: PendingSlot;
  progress: Progress;
}) {
  if (slot.live) {
    return (
      <StepPreview
        seq={progress.state === "generating" ? progress.preview_seq : 0}
        size={slot.entry.size}
        step={progress.state === "generating" ? progress.step : 0}
        total={progress.state === "generating" ? progress.total : 0}
      />
    );
  }
  const waiting = slot.entry.status === "queued" || slot.entry.status === "running";
  return (
    <div
      className={waiting ? "pg-preview" : "pg-preview pg-preview-ended"}
      style={{ aspectRatio: aspectOf(slot.entry.size) ?? "1 / 1" }}
    >
      {/* The same glow the preview uses, for the same reason: it is what says
          "still being made" about a box that will not change for a while. A run
          that has *ended* gets no animation — nothing is coming — and a mark
          instead, so a failed tile is not mistaken for one still loading. */}
      {waiting ? (
        <div className="pg-preview-glow" aria-hidden="true" />
      ) : (
        <span className="pg-preview-mark" aria-hidden="true">
          {slot.entry.status === "failed" ? "!" : "×"}
        </span>
      )}
    </div>
  );
}

/**
 * A run's state in words, and the way to stop it.
 *
 * One component for all four states rather than a caller-side switch, because
 * the states are exhaustive and every view has to answer all of them: a view
 * that drew "running" and quietly omitted "failed" would have the reported
 * defect back for the case that matters most.
 *
 * `cancelling` is the whole page's — the id of the generation whose cancel is in
 * flight, or null — so a caller passes what it has rather than comparing ids at
 * three call sites.
 */
export function PendingStatus({
  entry,
  progress,
  paused,
  cancelling,
  onCancel,
}: {
  entry: PlaygroundGeneration;
  progress: Progress;
  /** The queue is held: what is queued says so instead of pretending to wait its turn. */
  paused: boolean;
  cancelling: string | null;
  onCancel: (id: string) => void;
}) {
  switch (entry.status) {
    case "queued":
      return (
        <div className="pg-status">
          <span className="note">{paused ? "Held — the queue is paused." : "Queued…"}</span>
          <button
            className="small"
            disabled={cancelling === entry.id}
            onClick={() => onCancel(entry.id)}
          >
            Cancel
          </button>
        </div>
      );
    case "running":
      return (
        <Running
          entry={entry}
          progress={progress}
          onCancel={onCancel}
          cancelling={cancelling === entry.id}
        />
      );
    case "failed":
      return (
        <div className="notice notice-error" role="status">
          <strong>Generation failed.</strong> {entry.error}
        </div>
      );
    case "cancelled":
      return <p className="note">Cancelled.</p>;
    default:
      return null;
  }
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
 *
 * Nothing here is wired to a view: one server-side slot, addressed by `seq`,
 * which is why the same component serves the feed, the gallery and the light
 * table without a second progress subscription between them.
 */
export function StepPreview({
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

  // `WxH` as the request recorded it, through the same parse the gallery's tiles
  // reserve their boxes with; anything else falls back to a square box rather
  // than collapsing the placeholder to nothing.
  const ratio = aspectOf(size) ?? "1 / 1";

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
