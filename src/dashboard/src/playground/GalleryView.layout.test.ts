/**
 * The cascade facts the gallery is built on.
 *
 * Layout is a cascade property here, not a render property: jsdom has no layout
 * engine, so a render test cannot see where a tile landed. These are asked of
 * the real stylesheet (`css: true`), the way the sidebar-collapse and
 * popover-placement tests are.
 *
 * 1. The wall is a CSS Grid whose tracks are ADDED as the box widens, not
 *    stretched. This assertion has now been re-aimed twice, and the history is
 *    the point: it first asked for `column-count > 1` (satisfied by the
 *    `column-count: 4` bug — 600px tiles on a 2560px screen), then for
 *    `column-width: 320px` (satisfied by a multi-column wall that still read
 *    top-to-bottom). Both pinned a mechanism rather than the property, so both
 *    passed a wall laid out wrongly. `auto-fill` + a floor + a `1fr` ceiling is
 *    the property.
 * 2. Reading order is document order. Multicol fills a column before starting
 *    the next, so the visual order stopped matching the DOM from 1100px up, and
 *    at 1920px the newest tile — a run in flight, appended last — landed
 *    mid-wall. Grid fills row by row, so this holds by construction.
 * 3. The wall scrolls downwards, never sideways.
 * 4. A tile's layers stay inside the tile, under the floating composer.
 * 5. Spacing is owned by the grid's `gap`, not by a tile margin left over from
 *    the multi-column era (which would double the vertical gutter).
 * 6. A tile reserves its box before its picture loads, from the run's own
 *    `WxH`. This is what stops the wall re-flowing as intrinsic heights are
 *    decoded — at load, and on every frame of a window resize. It is also why
 *    no masonry library is needed: they exist to measure what is already known.
 * 7. The tile's toolbar is drawn *on* the tile, and the tile is what its
 *    absolute position resolves against.
 * 8. A placeholder tile is exactly as tall as the box it reserves, and is not
 *    hidden until hover: a control you have to find is bad, and a state you
 *    have to find is the defect the placeholder exists for.
 * 9. Motion is off under `prefers-reduced-motion: reduce`. jsdom does not
 *    evaluate media queries when it computes a style — measured: a
 *    `min-width: 900px` rule does not apply at its default 1024px width — so
 *    this one is asked of the CSSOM instead, which is the honest reading: the
 *    sheet contains the reduce block and it covers the selectors that move.
 */

import { describe, expect, it } from "vitest";

import "../styles.css";
import { aspectOf } from "./groups";

const GRID = `
  <section class="pg-studio">
    <div class="pg-gallery">
      <div class="pg-gallery-wall">
        <div class="pg-gallery-row" style="height: 200px">
        <div class="pg-gallery-cell" style="width: 360px">
        <figure class="pg-tile">
          <button class="pg-thumb"><img alt="" /></button>
          <div class="pg-image-actions">
            <div class="pg-image-tools" role="toolbar"></div>
          </div>
        </figure>
        </div>
        </div>
      </div>
    </div>
  </section>`;

/** A tile holding a run in flight instead of a picture. */
const PENDING = `
  <section class="pg-studio">
    <div class="pg-gallery">
      <div class="pg-gallery-wall">
        <div class="pg-gallery-row" style="height: 200px">
        <div class="pg-gallery-cell" style="width: 360px">
        <figure class="pg-tile pg-tile-pending">
          <div class="pg-preview" style="aspect-ratio: 1024 / 576"></div>
          <div class="pg-pending-note">
            <div class="pg-running"><div class="bar"></div><div class="pg-status"></div></div>
          </div>
        </figure>
        </div>
        </div>
      </div>
    </div>
  </section>`;

/** Every selector inside a `prefers-reduced-motion: reduce` block.
 *
 * Selector *groups* are split: a rule written for two elements at once is two
 * claims, and a test that had to spell the group exactly as the sheet writes it
 * would break on a comma. */
function reducedMotionSelectors(): string[] {
  const found: string[] = [];
  for (const sheet of document.styleSheets) {
    for (const rule of sheet.cssRules) {
      if (!(rule instanceof CSSMediaRule)) continue;
      if (!rule.conditionText.includes("prefers-reduced-motion")) continue;
      for (const inner of rule.cssRules) {
        if (inner instanceof CSSStyleRule) {
          for (const one of inner.selectorText.split(",")) found.push(one.trim());
        }
      }
    }
  }
  return found;
}

describe("the gallery's tiling", () => {
  it("lays the wall out as rows, each one spanning the full width", () => {
    document.body.innerHTML = GRID;
    const wall = getComputedStyle(document.querySelector(".pg-gallery-wall")!);
    const row = getComputedStyle(document.querySelector(".pg-gallery-row")!);
    const cell = getComputedStyle(document.querySelector(".pg-gallery-cell")!);

    // Re-aimed a third time, and the history is the argument. It first asked for
    // `column-count > 1` (true of the `column-count: 4` bug — 600px tiles on a
    // 2560px screen), then for `column-width: 320px` (true of a multicol wall
    // that still read top-to-bottom), then for a grid track list (true of a grid
    // of equal tracks, which gave every row the height of its tallest tile and
    // left a portrait-sized hole under every landscape beside it).
    //
    // Each of those pinned a *mechanism*. The property is that a row is a line
    // of pictures at one height and their own widths, scaled to span the wall —
    // and the scaling itself is arithmetic, proved in `justify.test.ts` where it
    // can be asserted exactly rather than guessed at through jsdom.
    expect(wall.display).toBe("flex");
    expect(wall.flexDirection).toBe("column");
    expect(row.display).toBe("flex");
    // One height per row, own width per cell: `stretch` is what makes a portrait
    // and a landscape the same height, which is what closes the hole.
    expect(row.alignItems).toBe("stretch");
    // Neither grow nor shrink: the solver already spent the width exactly, and
    // flex redistributing it would undo the aspect ratios it preserved.
    expect(cell.flexGrow).toBe("0");
    expect(cell.flexShrink).toBe("0");
    // It scrolls on its own, like the feed it replaces — the studio does not.
    expect(getComputedStyle(document.querySelector(".pg-gallery")!).overflowY).toBe("auto");
  });

  it("reads in document order, so the newest tile is the last one", () => {
    // The defect that ended the multi-column layout. Multicol filled a column
    // top-to-bottom before starting the next, so DOM order ran DOWN and the
    // visual order stopped matching it from 1100px up; at 1920px the newest tile
    // — a run in flight, appended last — landed mid-wall.
    //
    // Rows are laid out in order and each row's items in order, so document
    // order is reading order by construction. Asserted here as the two flex
    // directions that make it so: a `column-reverse` wall or a `row-reverse` row
    // would silently reverse the project.
    document.body.innerHTML = GRID;
    const wall = getComputedStyle(document.querySelector(".pg-gallery-wall")!);
    const row = getComputedStyle(document.querySelector(".pg-gallery-row")!);
    expect(wall.flexDirection).toBe("column");
    expect(row.flexDirection).not.toContain("reverse");
    // No `wrap` on a row either: the solver decides what fits, and a row that
    // wrapped would put a tile on a line the solver never sized.
    expect(row.flexWrap).not.toBe("wrap");
  });

  it("scrolls the wall downwards, never sideways", () => {
    // The reported bug, as a cascade claim. `.pg-gallery` is `flex: 1` inside a
    // flex column, so it has a *definite* height; a multi-column box with a
    // definite height cannot overflow downwards and lays its remainder out as
    // further columns to the RIGHT instead. While the two were one element the
    // wall therefore scrolled sideways, and a run in flight — appended last —
    // sat off the right edge behind a horizontal scrollbar. Measured in Chrome
    // at 1200x700 before the split: scrollWidth 2186 against clientWidth 872,
    // three tiles off-screen, verticallyScrollable false.
    //
    // jsdom has no layout engine, so this is asserted as the *structural*
    // property that makes the overflow direction right, which is what actually
    // regressed: the element that scrolls must not be the element that columns.
    document.body.innerHTML = GRID;
    const scroller = getComputedStyle(document.querySelector(".pg-gallery")!);
    const wall = getComputedStyle(document.querySelector(".pg-gallery-wall")!);

    // The scroll box does not column. jsdom reports an undeclared property as
    // `""` rather than resolving it to its initial value, so "not a length" is
    // the honest reading of "no columns here" — and it is exactly what fails if
    // the two roles are merged back onto one element.
    expect(["", "auto"]).toContain(scroller.columnWidth);
    expect(["", "auto"]).toContain(scroller.columnCount);
    // …and the wall declares no height and no flex of its own, so it grows
    // downwards into the scroll its parent provides. Same jsdom caveat: an
    // undeclared property reads as `""`, which is precisely the claim — the
    // wall must not be given the definite height that broke the scroll
    // direction. A `flex: 1` or `height: 100%` here reinstates the bug and
    // fails this test.
    expect(["", "0"]).toContain(wall.flexGrow);
    expect(["", "auto"]).toContain(wall.height);
    // The layout lives on the wall and nowhere else — the positive half of the
    // split, so a merge back onto one element cannot pass by declaring neither.
    expect(wall.display).toBe("flex");
    // Sideways scrolling is never the answer here, whatever the content does.
    expect(scroller.overflowX).not.toBe("scroll");
  });

  it("keeps a tile's layers inside the tile, under the floating composer", () => {
    // The reported defect: a pending run's status bar painted *over* the glass
    // composer. Its `z-index: 1` is a claim about the tile's own layers — the
    // tag under the toolbar under the note — but `position: relative` with no
    // `z-index` leaves the tile itself at `auto`, so those inner values ranked
    // against the page instead, and beat a dock that was also `auto` on
    // document order.
    //
    // Two halves, both asserted: the tile isolates, and the dock outranks it.
    document.body.innerHTML = GRID;
    expect(getComputedStyle(document.querySelector(".pg-tile")!).isolation).toBe("isolate");

    document.body.innerHTML = `<section class="pg-studio"><div class="pg-dock"></div></section>`;
    const dock = getComputedStyle(document.querySelector(".pg-dock")!);
    expect(dock.zIndex).toBe("10");
    // Below the popovers and the modal, which must still open over the
    // composer — a dock that outranked those would trade one defect for two.
    expect(Number(dock.zIndex)).toBeLessThan(20);
  });

  it("gives each tile exactly one cell, with the gap owning the spacing", () => {
    // Replaces "never splits a tile across a column boundary". `break-inside:
    // avoid` was a multi-column concern — a tile could be sliced across a column
    // break — and a grid item is never fragmented, so the old assertion now
    // pins a property that cannot fail and proves nothing.
    //
    // What can still fail is the spacing. Under multicol the tile's own
    // `margin-bottom` *was* the row gutter; leaving it under a grid that also
    // declares `gap` would double the vertical gutter against the horizontal
    // one. So: the wall owns the gap, the tile owns no margin.
    document.body.innerHTML = GRID;
    const wall = getComputedStyle(document.querySelector(".pg-gallery-wall")!);
    const tile = getComputedStyle(document.querySelector(".pg-tile")!);
    expect(wall.gap).not.toBe("");
    expect(wall.gap).not.toBe("normal");
    expect(tile.marginBottom).toBe("0px");
  });

  it("reserves a tile's box from the run's own size, before the picture loads", () => {
    // The flicker, at its source: `1024x576` is a shape the layout can be given
    // before a byte of the thumbnail has arrived.
    expect(aspectOf("1024x576")).toBe("1024 / 576");
    expect(aspectOf("512x512")).toBe("512 / 512");
    // And nothing guessed when the string is not two positive numbers: a wrong
    // ratio reserves a wrong box, which is a worse jump than a late one.
    expect(aspectOf("")).toBeNull();
    expect(aspectOf("1024")).toBeNull();
    expect(aspectOf("0x576")).toBeNull();
    expect(aspectOf("1024xabc")).toBeNull();
  });

  it("gives the tile's button the tile's width, so the reserved ratio resolves", () => {
    // A `<button>` is shrink-to-fit even at `display: block`, so `width: 100%`
    // on the image resolved against a box that was itself waiting for the
    // image. Measured in Chrome with the thumbnail requests aborted: 4×2 tiles
    // without this rule, 362×201 with it. The reserved `aspect-ratio` is worth
    // nothing unless the width it multiplies is known.
    document.body.innerHTML = GRID;
    expect(getComputedStyle(document.querySelector(".pg-tile .pg-thumb")!).width).toBe("100%");
    expect(getComputedStyle(document.querySelector(".pg-tile .pg-thumb img")!).width).toBe("100%");
  });

  it("draws the toolbar over the picture, anchored to the tile", () => {
    document.body.innerHTML = GRID;
    const actions = document.querySelector<HTMLElement>(".pg-tile .pg-image-actions")!;
    expect(getComputedStyle(actions).position).toBe("absolute");
    // Hidden until the tile is hovered or something inside it takes focus: the
    // point of this view is a wall of pictures with nothing drawn on them.
    expect(getComputedStyle(actions).opacity).toBe("0");
    // The tile clips, so the hover growth happens inside its corners rather
    // than over the tiles beside it.
    expect(getComputedStyle(document.querySelector(".pg-tile")!).overflow).toBe("hidden");
  });

  it("keeps a placeholder tile the height of the box it is holding", () => {
    // The other half of the reservation. The ratio makes the *box* the right
    // shape; this makes the *tile* the box, so the picture that replaces the
    // placeholder occupies the same cell and the columns are not re-balanced.
    // A status line in flow would add its own height to this tile and to no
    // other, which is the Step 7 flicker with an extra step.
    document.body.innerHTML = PENDING;
    const note = document.querySelector<HTMLElement>(".pg-tile-pending .pg-pending-note")!;
    expect(getComputedStyle(note).position).toBe("absolute");
    // And nothing dims it. jsdom's `getComputedStyle` reports only what the
    // matched rules declare, so this reads as "no rule hides this element",
    // which is the claim: the toolbar's `opacity: 0` above is what a state must
    // not inherit. A state you have to hover to find is the reported defect.
    expect(getComputedStyle(note).opacity).not.toBe("0");
    // The box itself spends the ratio it was given rather than an intrinsic
    // height it has no content to derive.
    const box = document.querySelector<HTMLElement>(".pg-tile-pending .pg-preview")!;
    expect(box.style.aspectRatio).toBe("1024 / 576");
    expect(getComputedStyle(box).width).toBe("100%");
  });

  it("moves nothing under prefers-reduced-motion: reduce", () => {
    const selectors = reducedMotionSelectors();
    // Every selector this pass gave a transform or a transition to has an
    // answer in the reduce block: the two tile hovers, the rail's collapse, and
    // the dialog's entrance.
    expect(selectors).toContain(".pg-tile:hover .pg-thumb img");
    expect(selectors).toContain(".pg-images .pg-thumb:hover img");
    expect(selectors).toContain(".pg-sidebar");
    expect(selectors).toContain(".modal-backdrop");
    expect(selectors).toContain(".modal");
  });
});
