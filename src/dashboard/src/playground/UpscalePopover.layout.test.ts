/**
 * Why the upscale popover needs a positioned ancestor.
 *
 * A cascade question, so it is asked of the real cascade (`css: true`) rather
 * than assumed.
 *
 * What went wrong: `.pg-popover` is `position: absolute` with
 * `bottom: calc(100% + …)`, so it paints above its containing block — which is
 * the nearest ancestor whose `position` is not `static`. The composer's copy
 * gets one from `.pg-advanced`. The feed's copy sat in `.pg-image-cell`, which
 * had no `position` at all, so it resolved against something far up the tree
 * and painted itself off the top of the page. The button toggled, the dialog
 * mounted, and nothing appeared.
 *
 * It is anchored to the toolbar's wrapper rather than the whole cell, so it
 * opens over the image: `.pg-feed` scrolls, and a panel hung above a 512px
 * cell is clipped whenever that cell is near the top of the view.
 *
 * The render tests could not see it: jsdom has no layout engine, so
 * `getByRole("dialog")` finds the panel wherever it is painted. `position` is a
 * computed style rather than a measurement, which is exactly why this one can
 * be pinned here.
 */

import { beforeEach, describe, expect, it } from "vitest";

import "../styles.css";

const CELL = `
  <div class="pg-images">
    <figure class="pg-image-cell">
      <button class="pg-thumb"><img alt="" /></button>
      <div class="pg-image-actions">
        <div class="pg-image-tools" role="toolbar">
          <button class="small pg-tool" id="trigger"></button>
        </div>
        <div class="pg-popover pg-upscale-pop" role="dialog"></div>
      </div>
    </figure>
  </div>`;

/** The element the popover's `bottom: 100%` actually resolves against.
 *
 * `position || "static"`, and the fallback is the whole point: jsdom returns an
 * empty string for a property no rule set, not the initial value. A plain
 * `!== "static"` therefore reports *every* ancestor as positioned, and this
 * helper answered "the image cell" whether or not the rule under test existed
 * — a check that looked like a witness and was not one.
 */
function containingBlock(element: Element): Element | null {
  let parent = element.parentElement;
  while (parent !== null) {
    const position = getComputedStyle(parent).position || "static";
    if (position !== "static") return parent;
    parent = parent.parentElement;
  }
  return null;
}

describe("the upscale popover's containing block", () => {
  beforeEach(() => {
    document.body.innerHTML = CELL;
  });

  it("is the toolbar's own wrapper, not something further up the tree", () => {
    const popover = document.querySelector(".pg-upscale-pop")!;
    expect(getComputedStyle(popover).position).toBe("absolute");
    expect(containingBlock(popover)?.className).toContain("pg-image-actions");
  });

  it("opens upward and aligns with the toolbar it belongs to", () => {
    // Left, unlike the composer's, whose gear sits in a right-hand cluster.
    const style = getComputedStyle(document.querySelector(".pg-upscale-pop")!);
    expect(style.left).toBe("0px");
    expect(style.right).toBe("auto");
  });

  it("draws above the feed rather than behind it", () => {
    // Below the modal's 50, above everything in the feed.
    const z = Number(getComputedStyle(document.querySelector(".pg-upscale-pop")!).zIndex);
    expect(z).toBeGreaterThan(0);
    expect(z).toBeLessThan(50);
  });
});
