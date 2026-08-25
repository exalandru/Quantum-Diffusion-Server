/**
 * What a project's generations amount to: lineages, and the images in them.
 *
 * Extracted from `GenerationFeed` when the gallery arrived, because the two
 * views must agree on *which* images a project has. Had the gallery flattened
 * `generations` on its own, "every image the feed shows appears in the gallery"
 * would be a coincidence maintained by hand in two places rather than a
 * consequence of one function — and the first divergence (an upscale joining its
 * source's lineage, say) would show in only one of them.
 */
import type { PlaygroundGeneration } from "../types";

/**
 * One feed entry: every generation of the same lineage, images together.
 *
 * `root` is the generation that opened the group and owns what the entry shows
 * of the request — prompt, model, size, steps, reference image. Its members keep
 * their own status, because each of them is still a separate run that can queue,
 * fail or be cancelled on its own.
 */
export type Group = {
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
  /**
   * The group's runs that have no picture — one slot per image still owed by a
   * queued or running member, one marker per member that failed or was
   * cancelled.
   *
   * Here rather than in each view for the reason `images` is here: "a
   * submission the server accepted is visible in every view" (T8) has to hold
   * by construction, and it cannot if the gallery and the light table each
   * decide on their own what a project has in flight. The feed does not read
   * this — it walks `members` and states each one's status under the entry it
   * belongs to — but the wall and the filmstrip have no such row to write on,
   * so they draw these instead, in this order, which is the feed's order.
   */
  pending: PendingSlot[];
};

/** One image, plus the generation that actually produced it. */
export type GroupImage = {
  url: string;
  seed: number;
  kind: PlaygroundGeneration["kind"];
  size: string;
  model: string;
  /**
   * The step count of the run that made *this* image, which is 0 for an
   * upscale: a super-resolution pass has no denoising loop, and the server
   * records `steps=0` for it (`admission.submit_upscale`). Carried here for the
   * same reason `size` and `model` are — the light table's inspector states the
   * settings of the selected image, and reading the root's step count would
   * attribute the original request's 6 steps to a picture no sampler produced.
   */
  steps: number;
};

/**
 * A run with no picture: what a view draws where a picture is not yet, or never.
 *
 * The whole record is carried rather than a copy of the fields a placeholder
 * needs. `size` reserves the box, `status` and `error` are the words, `id` is
 * what `onCancel` is called with, `n` counts the images the request asked for —
 * and a slot that copied four of those five would be a second, staler answer to
 * "what is this run" living beside the record the server sent.
 */
export type PendingSlot = {
  entry: PlaygroundGeneration;
  /**
   * The lineage's prompt, for the placeholder's accessible name.
   *
   * The root's, like every other prompt a view shows: an upscale has no prompt
   * of its own, and the server copies its source's into the record.
   */
  prompt: string;
  /**
   * Which image of the request this slot is waiting for, 1-based, or 0 for a
   * failed or cancelled marker — which stands for the run, not for one of its
   * images.
   */
  index: number;
  /**
   * The engine is denoising *this* slot now, so it carries the live preview.
   *
   * At most one slot of a project has it: the runner is a single FIFO worker,
   * the engine takes one job at a time, and within a job the images are made
   * one after another — so it is the first image a running member still owes.
   * An upscale never has it: a super-resolution pass decodes no intermediate
   * frames, which is why the feed skips the preview for one too.
   */
  live: boolean;
  /**
   * This slot speaks for the record: it is the one that states the run's status.
   *
   * A record has `n` owed boxes and exactly one status, so without this an `n=2`
   * request in flight drew "Image 1 of 2 — step 6 of 50" and a Cancel button on
   * *both* of its placeholders — two runs where there is one. Observed in the
   * browser, which is the only place a wall of two identical boxes is visible.
   * The first owed slot carries it; the others are boxes being held.
   */
  carriesStatus: boolean;
};

/**
 * The slots one member owes, in the order its images will land.
 *
 * `n - images.length` rather than `n`: a member of an `n=4` request that has
 * delivered two pictures owes two boxes, and a fourth placeholder standing next
 * to the picture that replaced it would be the wall claiming more is coming than
 * is. A failed or cancelled run owes nothing and gets one marker anyway — the
 * defect this exists for is a run *vanishing* from a view, and a run that ended
 * badly vanishing is the same defect wearing a different hat.
 */
const pendingOf = (entry: PlaygroundGeneration, prompt: string): PendingSlot[] => {
  if (entry.status === "queued" || entry.status === "running") {
    const owed = Math.max(0, entry.n - entry.images.length);
    return Array.from({ length: owed }, (_unused, at) => ({
      entry,
      prompt,
      index: entry.images.length + at + 1,
      live: entry.status === "running" && at === 0 && entry.kind !== "upscale",
      carriesStatus: at === 0,
    }));
  }
  if (entry.status === "failed" || entry.status === "cancelled") {
    return [{ entry, prompt, index: 0, live: false, carriesStatus: true }];
  }
  return [];
};

const imagesOf = (entry: PlaygroundGeneration): GroupImage[] =>
  entry.images.map((image) => ({
    ...image,
    kind: entry.kind,
    size: entry.size,
    model: entry.model,
    steps: entry.steps,
  }));

/** Groups in the order their roots appear: a group never moves as it grows. */
export function groupsOf(generations: PlaygroundGeneration[]): Group[] {
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
        pending: [],
      });
    } else {
      group.members.push(entry);
      group.images.push(...imagesOf(entry));
      group.requested += entry.n;
    }
  }
  // A second pass, because a slot's prompt is the *root's* and the root is only
  // known once the group exists — and because the slots of one group have to
  // read in member order whatever order the members were added in.
  for (const group of groups.values()) {
    group.pending = group.members.flatMap((member) => pendingOf(member, group.root.prompt));
  }
  return [...groups.values()];
}

/**
 * A tile's box, reserved before its picture has loaded.
 *
 * The gallery's tiles were `width: 100%; height: auto`, so the column balancer
 * had nothing to work with until every intrinsic height had been decoded: the
 * wall re-flowed as images arrived and again on every resize frame, which is
 * the flicker. `size` is the server's own `WxH` for the run that produced this
 * image — `width`/`height` are `NOT NULL` columns, and an upscale records
 * source × factor, so every kind carries one — and an `aspect-ratio` from it
 * makes the box known before the bytes are.
 *
 * `null` rather than a guessed default when the string is not two positive
 * numbers: a wrong ratio reserves the wrong box and *is* the jump, whereas
 * `height: auto` merely defers it to load time, which is where it was.
 */
export function aspectOf(size: string): string | null {
  const match = /^(\d+)x(\d+)$/.exec(size);
  if (match === null) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (width <= 0 || height <= 0) return null;
  return `${width} / ${height}`;
}

/**
 * An image's own facts, for the viewer's footer.
 *
 * The image's, not its root's: they differ for an upscale, which lives in its
 * source's entry. Shared by both views so the same picture reads the same way
 * whichever one it was opened from.
 */
export const detailOf = (image: GroupImage, nameOf: (id: string) => string): string =>
  image.kind === "upscale"
    ? `seed ${image.seed} · ${image.size} · upscaled`
    : `seed ${image.seed} · ${image.size} · ${nameOf(image.model)}`;
