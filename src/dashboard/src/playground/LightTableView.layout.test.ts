/**
 * The seven cascade facts the light table's three-part frame is built on.
 *
 * jsdom has no layout engine, so a render test cannot see whether the inspector
 * keeps its column or whether the strip scrolls. These are asked of the real
 * stylesheet (`css: true`), the way the gallery's tiling test and the
 * sidebar-collapse test are.
 *
 * 1. The frame is a two-track grid: the stage takes what is left, the inspector
 *    keeps a fixed width. `minmax(0, 1fr)` on the stage is load-bearing — with a
 *    plain `1fr` a 4096-pixel-wide picture sets the track's minimum and pushes
 *    the inspector off the edge.
 * 2. Both tracks fit inside the room the composer leaves. This one is a measured
 *    bug, not a precaution: with the row left implicit the frame overflowed its
 *    own padding box by 69px in the browser, which put the filmstrip and the
 *    inspector's last rows back under the floating composer. A definite row and
 *    the reservation on the frame — not on the two children — is the fix, and
 *    both halves of it are asserted because either one alone leaves the overlap.
 * 3. The filmstrip scrolls sideways and cannot be squeezed to nothing. Both are
 *    the strip's whole reason to exist: `flex: 0 0 auto` is what makes the stage
 *    give up height first when the studio is short.
 * 4. The settings list stacks one fact per row, against the shell's global
 *    two-column `dl` rule — which paired two facts per line here, because these
 *    `dt`/`dd` are wrapped rather than direct children. Also observed in a real
 *    browser, not anticipated.
 * 5. The nav buttons resolve against the stage, not against the studio. They are
 *    absolutely positioned, so without a positioned ancestor here they would
 *    drift over the project rail and the inspector.
 * 6. The inspector scrolls on its own. A prompt and an enhanced prompt together
 *    are taller than the stage, and a growing panel would push the filmstrip out
 *    of the studio.
 * 7. A placeholder frame in the strip is the strip's shape, not the run's. The
 *    box carries the run's own `aspect-ratio` inline — it is the same component
 *    the gallery's placeholder and the feed's cell use — so in the strip both of
 *    its axes are pinned by `inset` instead, which is what makes the browser
 *    ignore that ratio. Without it a portrait run in flight would make the whole
 *    row taller than the pictures beside it.
 */

import { beforeEach, describe, expect, it } from "vitest";

import "../styles.css";

const FRAME = `
  <section class="pg-studio">
    <div class="pg-table">
      <div class="pg-table-stage">
        <div class="pg-table-main">
          <button class="pg-table-nav prev"></button>
          <button class="pg-table-hero"><img alt="" /></button>
          <button class="pg-table-nav next"></button>
        </div>
        <div class="pg-strip" role="list">
          <button class="pg-strip-tile" aria-current="true"><img alt="" /></button>
          <button class="pg-strip-tile"><img alt="" /></button>
        </div>
      </div>
      <aside class="pg-inspector">
        <section class="pg-insp-section"><h3>Prompt</h3></section>
        <section class="pg-insp-section">
          <h3>Settings</h3>
          <dl class="pg-insp-facts">
            <div class="pg-kv"><dt>Model</dt><dd>sd35-large</dd></div>
            <div class="pg-kv"><dt>Size</dt><dd>1440x800</dd></div>
          </dl>
        </section>
      </aside>
    </div>
  </section>`;

describe("the light table's frame", () => {
  beforeEach(() => {
    document.body.innerHTML = FRAME;
  });

  it("gives the inspector a fixed column and the stage the rest", () => {
    const style = getComputedStyle(document.querySelector(".pg-table")!);
    expect(style.display).toBe("grid");
    // The stage track may shrink below its content; the panel may not shrink at
    // all. A `1fr 300px` would satisfy neither half of that.
    expect(style.gridTemplateColumns).toBe("minmax(0, 1fr) 300px");
  });

  it("keeps the whole frame out from under the floating composer", () => {
    const frame = getComputedStyle(document.querySelector(".pg-table")!);
    // Definite, not implicit: an `auto` row is sized from content the hero's
    // own `max-height: 100%` cannot bound yet, and the frame then overflows the
    // padding below it. Measured at 69px of overflow in a real browser.
    expect(frame.gridTemplateRows).toBe("minmax(0, 1fr)");
    // The reservation is the frame's, so it shortens both tracks at once. On
    // the two children instead, the grid row above them stays full height and
    // the padding buys nothing.
    expect(frame.paddingBottom).toContain("--pg-dock-h");
    // And it is not also spent inside the stage, which is where it was first
    // written and where it bought nothing.
    expect(
      getComputedStyle(document.querySelector(".pg-table-stage")!).paddingBottom,
    ).not.toContain("--pg-dock-h");
  });

  it("scrolls the filmstrip sideways without letting the stage crush it", () => {
    const style = getComputedStyle(document.querySelector(".pg-strip")!);
    expect(style.overflowX).toBe("auto");
    // Not `overflow: auto`: a vertical scrollbar on a one-row strip is a
    // scrollbar over a picture.
    expect(style.overflowY).toBe("hidden");
    expect(style.flexShrink).toBe("0");
    expect(style.flexGrow).toBe("0");
  });

  it("stacks the settings one fact per row", () => {
    // The shell styles every `dl` as an `auto 1fr` grid, which is right for a
    // panel whose `dt`/`dd` are direct children and wrong here: these pairs are
    // wrapped a row at a time, so under that rule two facts shared a line —
    // "Model sd35-large Size 1440x800" — with the labels no longer beside their
    // own values. Observed in the browser, so it is pinned here.
    expect(getComputedStyle(document.querySelector(".pg-insp-facts")!).display).toBe("block");
    const row = getComputedStyle(document.querySelector(".pg-kv")!);
    expect(row.display).toBe("flex");
    expect(row.justifyContent).toBe("space-between");
  });

  it("anchors the previous/next controls to the stage", () => {
    const nav = document.querySelector<HTMLElement>(".pg-table-nav")!;
    expect(getComputedStyle(nav).position).toBe("absolute");
    const stage = document.querySelector<HTMLElement>(".pg-table-main")!;
    expect(getComputedStyle(stage).position).toBe("relative");
    expect(nav.parentElement).toBe(stage);
  });

  it("scrolls the inspector on its own", () => {
    expect(getComputedStyle(document.querySelector(".pg-inspector")!).overflowY).toBe("auto");
  });

  it("gives a placeholder frame the strip's shape rather than the run's", () => {
    // The box is the gallery's and the feed's, and it carries the run's own
    // `WxH` inline — which is exactly right in a wall of pictures and wrong in a
    // row of them, where a portrait run in flight would set the height of every
    // frame beside it. Pinning both axes is what overrides the inline ratio, the
    // same way the strip's thumbnails are overridden by `object-fit: cover`.
    document.body.innerHTML = `
      <div class="pg-strip" role="list">
        <button class="pg-strip-tile"><img alt="" /></button>
        <button class="pg-strip-tile pg-strip-pending">
          <div class="pg-preview" style="aspect-ratio: 512 / 1024"></div>
        </button>
      </div>`;
    const frame = getComputedStyle(document.querySelector(".pg-strip-pending")!);
    expect(frame.aspectRatio).toBe("16 / 10");
    expect(frame.position).toBe("relative");
    // Not dimmed like the frames that are only context: a run in flight is the
    // one frame that is changing.
    expect(frame.opacity).toBe("1");
    const box = getComputedStyle(document.querySelector(".pg-strip-pending .pg-preview")!);
    expect(box.position).toBe("absolute");
    expect(box.inset).toBe("0");
  });
});

it("centres the picture even when its button is wider than it", () => {
  // Reported by the user, and missed by a probe that measured the wrong box.
  // `.pg-table-main` is a `place-items: center` grid, so the *button* is
  // centred and every measurement of the button came back symmetric — while
  // the picture inside it sat hard against the left edge, because a stretched
  // `<button>` puts its content at the inline start.
  //
  // Auto inline margins are what make the centring a property of the picture's
  // own box rather than of an assumption about its parent's width. Asserted on
  // both the button and the image: either one alone leaves the other free to
  // reintroduce the offset.
  document.body.innerHTML = FRAME;
  const button = getComputedStyle(document.querySelector(".pg-table-hero")!);
  const image = getComputedStyle(document.querySelector(".pg-table-hero img")!);

  // jsdom does not expand the `margin-inline` shorthand into its longhands, so
  // the shorthand itself is what can be read back.
  expect(button.marginInline).toBe("auto");
  expect(image.marginInline).toBe("auto");
});
