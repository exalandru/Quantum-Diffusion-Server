/**
 * The advanced settings panel is a centred rectangle in two columns.
 *
 * It was a 320px popover hung above the composer's gear: six fields in one
 * column, ~700px tall, opening upward from the bottom of the window. On a
 * laptop the top of it left the screen and there was nothing to scroll — the
 * panel was `position: absolute` in the composer, not a box with a height.
 *
 * Cascade questions, so they are asked of the real cascade (`css: true`), with
 * one exception noted below. jsdom does not evaluate media queries when it
 * computes a style — measured: a `min-width: 900px` rule does not apply at its
 * default 1024px width — so "one column when narrow" is asked of the CSSOM
 * instead. That is the honest reading of what can be established here: the
 * sheet carries the rule, in a `max-width` block, over the same selector.
 */

import { describe, expect, it } from "vitest";

import "../styles.css";

const DIALOG = `
  <div class="modal-backdrop">
    <div class="modal pg-settings" role="dialog">
      <div class="modal-head"><h3 class="modal-title">Advanced settings</h3></div>
      <div class="modal-body">
        <div class="pg-settings-grid">
          <div class="pg-field"></div>
          <div class="pg-field"></div>
        </div>
      </div>
    </div>
  </div>`;

function styleOf(selector: string): CSSStyleDeclaration {
  document.body.innerHTML = DIALOG;
  const element = document.querySelector(selector);
  if (!element) throw new Error(`no ${selector}`);
  return window.getComputedStyle(element);
}

/** The `grid-template-columns` this selector is given inside a media block. */
function inMediaBlocks(selector: string): { condition: string; columns: string }[] {
  const found: { condition: string; columns: string }[] = [];
  for (const sheet of document.styleSheets) {
    for (const rule of sheet.cssRules) {
      if (!(rule instanceof CSSMediaRule)) continue;
      for (const inner of rule.cssRules) {
        if (inner instanceof CSSStyleRule && inner.selectorText === selector) {
          found.push({
            condition: rule.conditionText,
            columns: inner.style.getPropertyValue("grid-template-columns"),
          });
        }
      }
    }
  }
  return found;
}

describe("the advanced settings dialog", () => {
  it("sits in the middle of the window rather than at the top of the page", () => {
    // `.modal-backdrop` starts every other dialog at `flex-start`, deliberately:
    // a form opened from a row belongs near the top. This one is a rectangle the
    // user asked to have centred, and `align-self` is how it says so without
    // moving every other dialog with it.
    document.body.innerHTML = DIALOG;
    expect(window.getComputedStyle(document.querySelector(".modal-backdrop")!).alignItems).toBe(
      "flex-start",
    );
    expect(styleOf(".modal.pg-settings").alignSelf).toBe("center");
  });

  it("lays its fields out in two columns at a roomy width", () => {
    expect(styleOf(".pg-settings-grid").gridTemplateColumns).toBe("1fr 1fr");
    // Wider than the sheet's 660px form measure, because two columns of fields
    // in 660px is two cramped columns.
    expect(styleOf(".modal.pg-settings").width).toBe("min(840px, 100%)");
  });

  it("falls back to one column when the window is narrow", () => {
    const narrow = inMediaBlocks(".pg-settings-grid");
    expect(narrow).toHaveLength(1);
    expect(narrow[0]!.condition).toContain("max-width");
    expect(narrow[0]!.columns).toBe("1fr");
  });

  it("can always be scrolled to its last field", () => {
    // The bug this replaces, guarded against in its new home: the dialog is
    // bounded by the viewport, so the body has to be the thing that scrolls or
    // the last field is unreachable on a short window all over again.
    expect(styleOf(".modal.pg-settings").maxHeight).toBe("calc(100vh - 2 * var(--s5))");
    expect(styleOf(".pg-settings .modal-body").overflowY).toBe("auto");
  });
});
