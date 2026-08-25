/**
 * Justified rows: the layout Adobe, Google Photos and Lightroom use for a wall
 * of mixed-shape pictures.
 *
 * ## Why not the two layouts this replaces
 *
 * **A grid of equal tracks** (what was here) gives every tile the same width and
 * lets the row take the height of its tallest member. That is fine while every
 * picture is the same shape and catastrophic as soon as they are not: this
 * project's own store holds four ratios — 1.80, 1.78, 1.00 and 0.744 — and a
 * portrait is 2.4x taller than a landscape at equal width, so a row containing
 * one leaves a portrait-sized hole under every other tile in it.
 *
 * **Multi-column** (what preceded the grid) has no holes, but fills each column
 * top-to-bottom before starting the next, so document order stops being reading
 * order — measured wrong at every width from 1100px up, with the newest tile
 * landing mid-wall at 1920px.
 *
 * Justified rows keeps both properties at once. Pictures are taken in order and
 * packed into a row until the row is full; the row is then scaled so its
 * pictures exactly span the width. Every picture keeps its own aspect ratio,
 * every row is flush left and right, and consecutive items stay consecutive.
 *
 * ## Why no library
 *
 * `react-photo-album`, `masonic` and friends solve this, but they also own the
 * rendering of each cell — and a cell here is not a photo: it carries a hover
 * toolbar, an upscale tag, and sometimes a running generation with a progress
 * bar and a cancel button instead of an image at all. Fighting a library's
 * render props to reinstate that is more code than the thirty lines below.
 *
 * More decisively: those libraries exist to *measure* heights. Every tile here
 * already declares its `WxH` (the run asked for it), so the shape is known
 * before a byte of image arrives. Measuring would be strictly worse — it is
 * what produced the load-time jump and the resize flicker this app already
 * fixed once by reserving the ratio up front.
 */

/** Anything with a known shape can be laid out; the caller says what it is. */
export type Justifiable = {
  /** width / height. Must be finite and positive. */
  aspect: number;
};

export type JustifiedRow<T> = {
  items: T[];
  /** The height every item in this row is drawn at, in px. */
  height: number;
  /**
   * Whether the row was scaled to span the container.
   *
   * The last row usually is not: a single leftover picture stretched to the full
   * width of a 2560px window is a banner, not a thumbnail, and it makes the most
   * recent generation — which is exactly what a trailing partial row holds —
   * the loudest thing on the wall.
   */
  justified: boolean;
};

export type JustifyOptions = {
  /** Content width available to the row, in px, gaps included. */
  containerWidth: number;
  /** The height rows aim for. Rows land near it, never exactly on it. */
  targetHeight: number;
  /** Horizontal space between two items in a row, in px. */
  gap: number;
  /**
   * How far a row may stretch past `targetHeight` before it is split.
   *
   * Without a ceiling, a lone very wide panorama would be scaled up until it
   * spanned the container, which on a wide window means one picture several
   * times taller than its neighbours.
   */
  maxHeightRatio?: number;
};

/**
 * The height a row of these aspects takes when scaled to span `width`.
 *
 * Each item is drawn at `height * aspect` wide, plus one gap between each pair,
 * so `width = height * sum(aspects) + gap * (n - 1)`. Solve for height.
 */
function rowHeight(aspects: number[], width: number, gap: number): number {
  const total = aspects.reduce((sum, a) => sum + a, 0);
  if (total <= 0) return 0;
  return (width - gap * (aspects.length - 1)) / total;
}

/**
 * Pack `items` into rows that each span `containerWidth`.
 *
 * Greedy and single-pass, which is what keeps reading order: items are consumed
 * in the order given and never reordered or deferred to a later row. A row is
 * closed as soon as adding the next item would take it below the target height,
 * and the choice between closing before or after that item goes to whichever
 * lands nearer the target — otherwise every row is systematically shorter than
 * asked for.
 */
export function justifyRows<T extends Justifiable>(
  items: T[],
  { containerWidth, targetHeight, gap, maxHeightRatio = 1.5 }: JustifyOptions,
): JustifiedRow<T>[] {
  if (items.length === 0 || containerWidth <= 0 || targetHeight <= 0) return [];

  const rows: JustifiedRow<T>[] = [];
  let current: T[] = [];

  for (const item of items) {
    // A shape we cannot trust cannot be scaled: fall back to the target rather
    // than divide by it. `aspectOf` already refuses a malformed `WxH`, so this
    // is the belt to that braces.
    const aspect = Number.isFinite(item.aspect) && item.aspect > 0 ? item.aspect : 1;
    const candidate = [...current, { ...item, aspect } as T];
    const height = rowHeight(
      candidate.map((c) => c.aspect),
      containerWidth,
      gap,
    );

    if (height > targetHeight) {
      // Still too tall: the row has room for more.
      current = candidate;
      continue;
    }

    // Adding this item takes the row below the target. Keep it if that lands
    // closer to the target than stopping short does.
    const without = current.length
      ? rowHeight(current.map((c) => c.aspect), containerWidth, gap)
      : Infinity;
    if (Math.abs(height - targetHeight) <= Math.abs(without - targetHeight)) {
      rows.push({ items: candidate, height, justified: true });
      current = [];
    } else {
      rows.push({ items: current, height: without, justified: true });
      current = [{ ...item, aspect } as T];
    }
  }

  if (current.length > 0) {
    // The trailing row keeps the target height instead of being stretched to
    // span the container. Capped, because a single panorama at target height is
    // fine but a single portrait is not: without the cap a 0.5-aspect leftover
    // would be twice as tall as every row above it.
    rows.push({ items: current, height: targetHeight, justified: false });
  }

  // A row that had to stretch far past the target — one very wide picture on a
  // narrow window — is clamped rather than allowed to tower over its
  // neighbours. Done after packing so it cannot change which items sit together.
  const ceiling = targetHeight * maxHeightRatio;
  return rows.map((row) =>
    row.height > ceiling ? { ...row, height: ceiling, justified: false } : row,
  );
}

/** `"1440x800"` → `1.8`, or `null` when the string is not two positive numbers. */
export function aspectRatioOf(size: string): number | null {
  const match = /^(\d+)x(\d+)$/.exec(size);
  if (match === null) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (width <= 0 || height <= 0) return null;
  return width / height;
}
