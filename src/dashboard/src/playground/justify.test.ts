import { describe, expect, it } from "vitest";

import { aspectRatioOf, justifyRows } from "./justify";

/**
 * The four aspect ratios this project's own store actually holds, measured:
 * 1.80 (1440x800), 1.778 (1280x720), 1.00 (512x512) and 0.744 (1072x1440).
 *
 * The mix is the whole point. A grid of equal tracks handles the first two and
 * falls apart on the last two — a portrait is 2.4x taller than a landscape at
 * equal width, so it leaves a portrait-sized hole under every other tile in its
 * row. That is the reported defect, and these are the shapes that produce it.
 */
const LANDSCAPE = 1.8;
const WIDE = 1.778;
const SQUARE = 1;
const PORTRAIT = 0.744;

const OPTS = { containerWidth: 1200, targetHeight: 240, gap: 10 };

/** What the row actually spans once every item is drawn at the row's height. */
function spanOf(row: { items: { aspect: number }[]; height: number }, gap: number) {
  const pictures = row.items.reduce((sum, i) => sum + i.aspect * row.height, 0);
  return pictures + gap * (row.items.length - 1);
}

describe("justifyRows", () => {
  it("makes every justified row span the container exactly", () => {
    // The property that removes the holes. Each picture keeps its own ratio and
    // the row is scaled until the pictures plus the gaps fill the width — so
    // there is no leftover space to leave a gap in.
    const items = [
      LANDSCAPE, PORTRAIT, SQUARE, WIDE, LANDSCAPE, PORTRAIT, SQUARE, WIDE,
      LANDSCAPE, PORTRAIT, SQUARE, WIDE,
    ].map((aspect) => ({ aspect }));

    const rows = justifyRows(items, OPTS);
    const justified = rows.filter((r) => r.justified);
    expect(justified.length).toBeGreaterThan(1);
    for (const row of justified) {
      expect(spanOf(row, OPTS.gap)).toBeCloseTo(OPTS.containerWidth, 6);
    }
  });

  it("keeps every item, once, in the order given", () => {
    // Reading order is the property multi-column lost and this must not lose
    // again: a project reads chronologically, and the newest tile — a run in
    // flight — is appended last and must stay last.
    const cycle = [LANDSCAPE, PORTRAIT, SQUARE, WIDE];
    const items = Array.from({ length: 23 }, (_, i) => ({
      aspect: cycle[i % 4] as number,
      id: i,
    }));

    const flat = justifyRows(items, OPTS).flatMap((r) => r.items);
    expect(flat.map((i) => i.id)).toEqual(items.map((i) => i.id));
  });

  it("draws a portrait and a landscape at the same height in one row", () => {
    // The defect, stated directly. Equal *tracks* give both the same width, so
    // the portrait is far taller and its row grows to fit it. Equal *height* is
    // what a justified row gives instead, and it is why the row has no hole.
    const rows = justifyRows(
      [{ aspect: PORTRAIT }, { aspect: LANDSCAPE }, { aspect: SQUARE }, { aspect: WIDE }],
      { ...OPTS, targetHeight: 200 },
    );
    const first = rows[0]!;
    expect(first.items.length).toBeGreaterThan(1);
    // One height for the row; the widths differ instead.
    const widths = first.items.map((i) => i.aspect * first.height);
    expect(new Set(widths.map((w) => Math.round(w))).size).toBeGreaterThan(1);
    expect(first.height).toBeGreaterThan(0);
  });

  it("does not stretch the trailing row across the window", () => {
    // A single leftover picture spanning 2560px is a banner, not a thumbnail —
    // and a trailing partial row is exactly where the newest generation sits.
    const rows = justifyRows(
      [{ aspect: LANDSCAPE }, { aspect: LANDSCAPE }, { aspect: LANDSCAPE }, { aspect: SQUARE }],
      { containerWidth: 2560, targetHeight: 240, gap: 10 },
    );
    const last = rows[rows.length - 1]!;
    expect(last.justified).toBe(false);
    expect(last.height).toBeLessThanOrEqual(240 * 1.5);
  });

  it("clamps a row that would tower over its neighbours", () => {
    // One very wide picture on a narrow window: scaled to span, it would be
    // several times taller than every row above it.
    const rows = justifyRows([{ aspect: 0.4 }], {
      containerWidth: 1200, targetHeight: 200, gap: 10, maxHeightRatio: 1.5,
    });
    expect(rows[0]!.height).toBeLessThanOrEqual(300);
  });

  it("survives a shape it cannot trust", () => {
    // `aspectOf` refuses a malformed `WxH`, but a zero or a NaN reaching this
    // must not divide the layout by it and blank the wall.
    const rows = justifyRows(
      [{ aspect: Number.NaN }, { aspect: 0 }, { aspect: -3 }, { aspect: LANDSCAPE }],
      OPTS,
    );
    const flat = rows.flatMap((r) => r.items);
    expect(flat).toHaveLength(4);
    for (const row of rows) {
      expect(Number.isFinite(row.height)).toBe(true);
      expect(row.height).toBeGreaterThan(0);
    }
  });

  it("returns nothing for nothing, and for a container with no width", () => {
    expect(justifyRows([], OPTS)).toEqual([]);
    expect(justifyRows([{ aspect: LANDSCAPE }], { ...OPTS, containerWidth: 0 })).toEqual([]);
  });

  it("reads a size string, and refuses one it cannot parse", () => {
    expect(aspectRatioOf("1440x800")).toBeCloseTo(1.8, 6);
    expect(aspectRatioOf("1072x1440")).toBeCloseTo(0.744, 3);
    expect(aspectRatioOf("512x512")).toBe(1);
    expect(aspectRatioOf("")).toBeNull();
    expect(aspectRatioOf("1440")).toBeNull();
    expect(aspectRatioOf("0x800")).toBeNull();
    expect(aspectRatioOf("1440xabc")).toBeNull();
  });
});
