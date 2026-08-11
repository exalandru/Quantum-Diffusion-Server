/** The Logs view fills the window instead of sitting in a fixed card.
 *
 * A stylesheet test rather than a render test, and the reason bounds what it
 * proves. jsdom has no layout engine: it parses CSS and answers `matches()`, but
 * it computes no boxes, so "the pane is 700px tall" is not observable here. What
 * *is* observable — and what actually breaks — is the cascade. The console used
 * to be a card with `max-height: calc(100vh - 230px)` written inline; the
 * redesign replaced that with a flex chain, and a single surviving height cap
 * anywhere in the sheet silently out-ranks `flex-grow` and puts the fixed box
 * back. Every review that reads only the newer rule sees a working layout.
 *
 * So the property pinned here is the one that fails: **no rule that applies to
 * the Logs console may cap its height, and its container must be anchored to the
 * viewport.** Both are checked against the real stylesheet, through the real
 * markup the component renders.
 *
 * What this does not establish: that the rendered pane is actually tall. That
 * needs a browser, and it belongs to the packaged visual witnesses.
 */

import { describe, expect, it } from "vitest";

// The application's own stylesheet, imported exactly as `main.tsx` imports it,
// so what is examined is the parsed cascade the app really gets rather than a
// copy of the text. `css: true` in `vitest.config.ts` is what makes that real;
// without it Vitest stubs the import and every assertion here would pass while
// checking an empty sheet — which the `styleRules` guard below catches.
import "./styles.css";

/** Declarations that cap a box's height. Any one of them defeats `flex-grow`. */
const CAPPING = ["max-height", "height"];

/** The Logs view's markup, as `App.tsx` and `Logs.tsx` render it together. */
function logsConsole(): HTMLElement {
  document.body.innerHTML = `
    <main class="view-logs">
      <header></header>
      <section class="panel logs">
        <div class="log-toolbar"></div>
        <div class="console"><div class="line">a line</div></div>
      </section>
    </main>`;
  const element = document.querySelector("main.view-logs > section.panel.logs > div.console");
  if (!(element instanceof HTMLElement)) throw new Error("the Logs markup changed shape");
  return element;
}

/** Every style rule the document carries, read through the CSSOM rather than by
 *  regex, so a rule split across lines or ordered differently is still seen. */
function styleRules(): { selector: string; style: CSSStyleDeclaration }[] {
  const rules: { selector: string; style: CSSStyleDeclaration }[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    for (const rule of Array.from(sheet.cssRules)) {
      if (rule instanceof CSSStyleRule) rules.push({ selector: rule.selectorText, style: rule.style });
    }
  }
  if (rules.length === 0) throw new Error("no stylesheet reached the document");
  return rules;
}

function applying(element: HTMLElement) {
  return styleRules().filter(({ selector }) => {
    // A selector jsdom cannot parse cannot be shown to apply, and a rule that
    // does not apply is not this test's business.
    try {
      return element.matches(selector);
    } catch {
      return false;
    }
  });
}

describe("the Logs pane is not capped to a fixed height", () => {
  it("has no rule that caps the height of its console", () => {
    const element = logsConsole();
    const capping = applying(element).filter(({ style }) =>
      CAPPING.some((property) => style.getPropertyValue(property) !== ""),
    );
    expect(capping.map(({ selector }) => selector)).toEqual([]);
  });

  it("lets the console grow and shrink inside the flex chain", () => {
    const element = logsConsole();
    const declared = (property: string) =>
      applying(element)
        .map(({ style }) => style.getPropertyValue(property))
        .filter((value) => value !== "");

    // `flex: 1 1 auto` is what claims the leftover space, and `overflow: auto` is
    // what makes the overflow scroll inside the pane instead of pushing the page.
    expect(declared("flex").join(" ")).toContain("1 1 auto");
    expect(declared("overflow").join(" ")).toContain("auto");
  });

  it("anchors the view to the viewport with a definite height", () => {
    // `min-height: 100vh` looks equivalent and is not: with a floor rather than a
    // definite height the flex chain has nothing to distribute, `main` grows to
    // the whole log, and the page scrolls while the pane sits at content height.
    document.body.innerHTML = `<main class="view-logs"></main>`;
    const main = document.querySelector("main.view-logs") as HTMLElement;
    const heights = applying(main)
      .map(({ style }) => style.getPropertyValue("height"))
      .filter((value) => value !== "");
    expect(heights).toContain("100vh");

    const minHeights = applying(main)
      .map(({ style }) => style.getPropertyValue("min-height"))
      .filter((value) => value !== "");
    expect(minHeights).toEqual([]);
  });

  it("would notice a stubbed stylesheet", () => {
    // The counterexample: every assertion above passes trivially against an empty
    // sheet, so the sheet's presence is itself checked.
    expect(styleRules().length).toBeGreaterThan(20);
  });
});
