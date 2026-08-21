/** The preview blur curve.
 *
 * A pure function over a small domain, so it is asserted directly rather than
 * through the DOM — the same reason `AdvancedSettings.test.ts` tests `sizeOf`
 * that way. What matters is not the exact pixel counts but the three properties
 * the curve exists for: it starts opaque, it always eases *down*, and a short
 * schedule stays softer at the end than a long one.
 */

import { describe, expect, it } from "vitest";

import { MAX_BLUR_PX, floorFor, previewBlurPx, previewScale } from "./preview-blur";

describe("previewBlurPx", () => {
  it("starts at the full blur, whatever the schedule", () => {
    for (const total of [8, 20, 30, 50]) {
      expect(previewBlurPx(0, total)).toBeCloseTo(MAX_BLUR_PX, 6);
    }
  });

  it("never increases as the run progresses", () => {
    for (const total of [8, 12, 30, 50]) {
      let previous = Infinity;
      for (let step = 0; step <= total; step++) {
        const blur = previewBlurPx(step, total);
        expect(blur).toBeLessThanOrEqual(previous);
        previous = blur;
      }
    }
  });

  it("keeps a short run softer at the end than a long one", () => {
    // The whole reason the floor depends on `total`: the last step of an 8-step
    // schedule is a much coarser image than the last step of a 50-step one.
    expect(previewBlurPx(8, 8)).toBeGreaterThan(previewBlurPx(50, 50));
    expect(previewBlurPx(8, 8)).toBeGreaterThan(previewBlurPx(30, 30));
    expect(previewBlurPx(50, 50)).toBeLessThan(2);
  });

  it("lands on its floor at the last step", () => {
    for (const total of [8, 30, 50]) {
      expect(previewBlurPx(total, total)).toBeCloseTo(floorFor(total), 6);
    }
  });

  it("is bounded by the floor and the maximum everywhere", () => {
    for (const total of [1, 8, 50, 100]) {
      for (let step = 0; step <= total; step++) {
        const blur = previewBlurPx(step, total);
        expect(blur).toBeGreaterThanOrEqual(floorFor(total));
        expect(blur).toBeLessThanOrEqual(MAX_BLUR_PX);
      }
    }
  });

  it("falls back to the full blur when there is no step count to go on", () => {
    // Weights still loading, or an external `/v1` client holds the engine: a
    // guessed denominator would say less than saying nothing.
    expect(previewBlurPx(0, 0)).toBe(MAX_BLUR_PX);
    expect(previewBlurPx(5, 0)).toBe(MAX_BLUR_PX);
    expect(previewBlurPx(5, -1)).toBe(MAX_BLUR_PX);
    expect(previewBlurPx(5, NaN)).toBe(MAX_BLUR_PX);
  });

  it("clamps a step count past the total", () => {
    expect(previewBlurPx(99, 30)).toBeCloseTo(floorFor(30), 6);
  });
});

describe("floorFor", () => {
  it("is clamped at both ends", () => {
    expect(floorFor(1)).toBe(6); // 48/1, clamped down
    expect(floorFor(1000)).toBe(0.5); // 48/1000, clamped up
  });

  it("decreases with the length of the schedule", () => {
    expect(floorFor(8)).toBeGreaterThan(floorFor(30));
    expect(floorFor(30)).toBeGreaterThan(floorFor(50));
  });
});

describe("previewScale", () => {
  it("inflates most when the blur is widest", () => {
    // `.pg-preview` clips, and a blur samples past the edge: without this the
    // container's background bleeds in around a blurred frame.
    expect(previewScale(MAX_BLUR_PX)).toBeGreaterThan(previewScale(1));
    expect(previewScale(0)).toBe(1);
  });
});
