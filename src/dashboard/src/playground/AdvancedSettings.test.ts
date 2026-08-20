/** The two pure functions the advanced popover rests on.
 *
 * `sizeOf` is the contract with the server — the chosen resolution is the *long*
 * side, whichever way the image stands, and an `auto` ratio sends nothing at all
 * so the model keeps its own default. `ratioLabel` is the contract with the eye:
 * the stored key is landscape-first, so portrait has to read back reversed.
 *
 * Both are covered here rather than through the DOM because both are total
 * functions over a small domain, and the whole domain is cheap to assert.
 */

import { describe, expect, it } from "vitest";

import {
  DEFAULT_ADVANCED,
  RATIOS,
  RESOLUTIONS,
  ratioLabel,
  sizeOf,
  type Advanced,
} from "./AdvancedSettings";

const at = (over: Partial<Advanced>): Advanced => ({ ...DEFAULT_ADVANCED, ...over });

describe("sizeOf", () => {
  it("sends nothing while the ratio is auto", () => {
    expect(sizeOf(DEFAULT_ADVANCED)).toBeNull();
    // Steps and seed are independent: an auto ratio still means "model default
    // size", never a size derived from the untouched resolution.
    expect(sizeOf(at({ steps: 8, seed: 42, long: 3840 }))).toBeNull();
  });

  it("spends the chosen resolution on the long side", () => {
    expect(sizeOf(at({ ratio: "16:9", long: 1920 }))).toBe("1920x1080");
    expect(sizeOf(at({ ratio: "16:9", long: 1920, orientation: "portrait" }))).toBe("1080x1920");
    expect(sizeOf(at({ ratio: "5:4", long: 2048 }))).toBe("2048x1638");
    expect(sizeOf(at({ ratio: "3:2", long: 512, orientation: "portrait" }))).toBe("341x512");
    expect(sizeOf(at({ ratio: "21:9", long: 3840 }))).toBe("3840x1646");
  });

  it("leaves a square square", () => {
    for (const orientation of ["landscape", "portrait"] as const) {
      expect(sizeOf(at({ ratio: "1:1", long: 1024, orientation }))).toBe("1024x1024");
    }
  });

  it("never puts the long side second in landscape, nor first in portrait", () => {
    for (const ratio of Object.keys(RATIOS) as (keyof typeof RATIOS)[]) {
      for (const long of RESOLUTIONS) {
        const [w, h] = sizeOf(at({ ratio, long }))!.split("x").map(Number);
        expect(w).toBe(long);
        expect(w).toBeGreaterThanOrEqual(h);

        const [pw, ph] = sizeOf(at({ ratio, long, orientation: "portrait" }))!
          .split("x")
          .map(Number);
        expect(ph).toBe(long);
        expect(ph).toBeGreaterThanOrEqual(pw);
      }
    }
  });
});

describe("ratioLabel", () => {
  it("reads landscape-first as stored", () => {
    for (const ratio of Object.keys(RATIOS) as (keyof typeof RATIOS)[]) {
      expect(ratioLabel(ratio, "landscape")).toBe(ratio);
    }
  });

  it("reverses every non-square ratio in portrait", () => {
    expect(ratioLabel("16:9", "portrait")).toBe("9:16");
    expect(ratioLabel("21:9", "portrait")).toBe("9:21");
    expect(ratioLabel("3:2", "portrait")).toBe("2:3");
    expect(ratioLabel("4:3", "portrait")).toBe("3:4");
    expect(ratioLabel("5:4", "portrait")).toBe("4:5");
    // A square has no other way round to name.
    expect(ratioLabel("1:1", "portrait")).toBe("1:1");
  });
});
