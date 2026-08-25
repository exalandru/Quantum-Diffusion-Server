/**
 * Where the upscale panel is drawn, and why it can no longer be cut in half.
 *
 * This file was three assertions about a stylesheet-positioned panel: that its
 * containing block was `.pg-image-actions`, that it sat at `left: 0`, and that
 * its `z-index` was below the modal's 50. All three described the mechanism
 * Step 1 chose — `position: absolute; bottom: calc(100% + …)` inside the
 * toolbar's wrapper — and all three were true of an implementation that had a
 * bug the test could not see: `.pg-feed` and `.pg-gallery` are `overflow-y:
 * auto`, and a scroll container clips whatever its descendants paint outside
 * its box. A panel anchored inside one is cut off whenever the tile it belongs
 * to sits near the top of the view, and the remainder is painted lower down the
 * page. That is the user's screenshot.
 *
 * So the assertions are re-aimed at the mechanism that replaced it, not
 * loosened. Each one is a claim about the *bug*, where the old ones were claims
 * about a coordinate:
 *
 *  - "containing block is `.pg-image-actions`" → the panel has **no clipping
 *    ancestor at all**: it is portalled to the body, so nothing between it and
 *    the document can cut it. This is strictly stronger, and it is the property
 *    the screenshot violated. Under the old implementation the panel's parent
 *    chain ran through `.pg-gallery`, so this test fails against it.
 *  - "`left: 0`, `right: auto`" → the panel is **placed against its trigger and
 *    flips** when there is no room above it. `left: 0` said "aligned with the
 *    toolbar" in a world where the sheet did the aligning; RAC measures the
 *    trigger now, and the falsifiable claim is the flip, which is what makes
 *    the panel reachable at the top edge of the scroll box.
 *  - "`z-index` below the modal's 50" → the panel **spends the height RAC gives
 *    it**: `overflow-y: auto` on the chrome, so a panel bounded to the viewport
 *    can still be scrolled to its Upscale button. The z-index claim died with
 *    the sheet's positioning — RAC writes `z-index: 100000` inline on a body
 *    portal, and a portalled panel is above the modal because it is *later in
 *    the document*, not because this sheet said so. Asserting the new number
 *    would be asserting react-aria's constant, which is not this project's
 *    contract; asserting that the panel is usable when bounded is.
 *
 * jsdom has no layout engine, so the placement half is measured by stubbing the
 * three inputs `calculatePosition` actually reads — `getBoundingClientRect`,
 * `offsetWidth`/`offsetHeight`, and the document element's client box. That is
 * not a mock of the answer: RAC's real collision arithmetic runs on the
 * measurements it is given, and the two cases below differ only in where the
 * trigger sits.
 */

import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import "../styles.css";
import type { Upscaler } from "../types";
import { UpscalePopover } from "./UpscalePopover";

const UPSCALERS: Upscaler[] = [
  {
    id: "realesrgan-x4plus",
    name: "Real-ESRGAN ×4",
    scales: [2, 4],
    downloaded: true,
    sizeMb: 64,
    license: "BSD-3-Clause",
  },
];

/** The panel's height, and the toolbar button's top edge, as RAC will read them. */
function measure(triggerTop: number, panelHeight: number) {
  Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
    configurable: true,
    get(this: HTMLElement) {
      return this.classList.contains("pg-popover") ? panelHeight : 30;
    },
  });
  Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
    configurable: true,
    get() {
      return 260;
    },
  });
  // The boundary RAC collides against: jsdom reports 0 for all of these, and a
  // zero-sized viewport makes every placement fit.
  for (const [property, value] of [
    ["clientHeight", 800],
    ["clientWidth", 1200],
    ["scrollHeight", 800],
    ["scrollWidth", 1200],
  ] as const) {
    Object.defineProperty(document.documentElement, property, {
      configurable: true,
      get: () => value,
    });
  }
  Element.prototype.getBoundingClientRect = function (this: Element) {
    if (this.tagName === "BUTTON") {
      return new DOMRect(40, triggerTop, 120, 30);
    }
    return new DOMRect(0, 0, 1200, 800);
  };
}

/** A tile's Upscale button inside the gallery's scroll box, and the open panel. */
function open(triggerTop: number, panelHeight = 300): HTMLElement {
  const gallery = document.createElement("div");
  gallery.className = "pg-gallery";
  const tile = document.createElement("figure");
  tile.className = "pg-tile";
  const button = document.createElement("button");
  tile.append(button);
  gallery.append(tile);
  document.body.append(gallery);

  measure(triggerTop, panelHeight);
  render(
    createElement(UpscalePopover, {
      upscalers: UPSCALERS,
      choice: { model: UPSCALERS[0]!.id, scale: 2 },
      onChoose: vi.fn(),
      onSubmit: vi.fn(),
      onClose: vi.fn(),
      trigger: { current: button },
    }),
    { container: tile.appendChild(document.createElement("div")) },
  );
  // The positioned wrapper, which is the panel: the `role="dialog"` element is
  // the fields inside it.
  return screen.getByRole("dialog", { name: "Upscale options" }).parentElement!;
}

const rect = Element.prototype.getBoundingClientRect;

beforeEach(() => {
  document.body.innerHTML = "";
});

afterEach(() => {
  Element.prototype.getBoundingClientRect = rect;
});

describe("the upscale panel", () => {
  it("has no ancestor that could clip it", () => {
    const panel = open(600);

    // The bug, stated as a property. `.pg-gallery` scrolls; so does `.pg-feed`;
    // so does the light table's inspector. The panel must be outside all of
    // them, and the only way to be sure of that without naming them is to walk
    // what it is actually inside.
    const clipping: string[] = [];
    for (let node = panel.parentElement; node !== null; node = node.parentElement) {
      const style = getComputedStyle(node);
      for (const overflow of [style.overflow, style.overflowX, style.overflowY]) {
        if (overflow === "auto" || overflow === "scroll" || overflow === "hidden") {
          clipping.push(`${node.tagName}.${node.className}: ${overflow}`);
        }
      }
    }
    expect(clipping).toEqual([]);
    // And said the other way round, because "no clipping ancestor" would also
    // be true of a panel that never mounted: it is a child of the document, not
    // of the tile that opened it.
    expect(panel.closest(".pg-gallery")).toBeNull();
    expect(panel.parentElement?.parentElement).toBe(document.body);
  });

  it("opens above the trigger when there is room above it", () => {
    expect(open(600).getAttribute("data-placement")).toBe("top");
  });

  it("flips below the trigger when there is not", () => {
    // The tile at the top edge of the scroll box: 20px down the viewport, with a
    // 300px panel asked to open upward. Upward is off the top of the window, so
    // the only correct answer is downward — and the old implementation had no
    // way to give it, because `bottom: calc(100% + …)` is not a decision.
    const panel = open(20);
    expect(panel.getAttribute("data-placement")).toBe("bottom");
    // Placed, not merely labelled: the panel starts below the trigger's bottom
    // edge (20 + 30 + the 10px offset).
    expect(panel.style.top).toBe("60px");
  });

  it("scrolls its own content, so a viewport-bounded panel stays usable", () => {
    // RAC writes `max-height` when the panel does not fit. Without this the
    // Upscale button below the fold could not be reached — the clipping bug
    // again, one element further in.
    expect(getComputedStyle(open(600)).overflowY).toBe("auto");
  });
});
