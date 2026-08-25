/**
 * The playground masthead's right-hand group, and the rail footer that took the
 * other half of it.
 *
 * A cascade question, so it is asked of the cascade (`css: true` in the vitest
 * config) rather than assumed: `.pg-pause` and `.shell-link` each carry an
 * `auto` left margin, and two of those in one flex row split the free space
 * between them instead of grouping at the edge — which left the button stranded
 * in the middle of the header.
 *
 * Re-aimed when the rail became flush and full height. The masthead is the
 * stage's now, not the page's — `main.playground` is a row holding the rail and
 * the stage, and the header is inside the stage — so the selector under test is
 * `.pg-stage > header`. And the way out of the page moved into the rail's
 * footer strip, so the second `auto` margin is cancelled *there* rather than by
 * a sibling rule in the header. Same hazard, same property, at the element that
 * now has it: `.shell-link` still ships `margin-left: auto`, and a container
 * that places its children with `justify-content` must overrule it or the pill
 * and the link stop being a strip.
 */

import { describe, expect, it } from "vitest";

import "../styles.css";

function styleOf(html: string, selector: string): CSSStyleDeclaration {
  document.body.innerHTML = html;
  const element = document.querySelector(selector);
  if (!element) throw new Error(`no ${selector}`);
  return window.getComputedStyle(element);
}

const HEADER = `
  <main class="playground">
    <div class="pg-layout">
      <div class="pg-stage">
        <header>
          <nav class="views"><button class="view-tab">Prompts</button></nav>
          <button class="small pg-pause">Pause queue</button>
        </header>
      </div>
    </div>
  </main>`;

const RAIL = `
  <aside class="pg-sidebar">
    <div class="pg-rail-foot">
      <span class="pill pill-live">Running</span>
      <a class="shell-link" href="/dashboard/">Server Config</a>
    </div>
  </aside>`;

describe("the playground masthead", () => {
  it("lays its controls out in a row, so nothing stretches", () => {
    // The shell's own header is a column; the playground's overrides it. A
    // button in a column flex row would span the whole header.
    expect(styleOf(HEADER, ".pg-stage > header").flexDirection).toBe("row");
  });

  it("pushes the pause control to the far edge", () => {
    expect(styleOf(HEADER, ".pg-pause").marginLeft).toBe("auto");
  });
});

describe("the rail's footer strip", () => {
  it("cancels the shell link's own auto margin, so the strip stays a strip", () => {
    expect(styleOf(RAIL, ".pg-rail-foot .shell-link").marginLeft).toBe("0px");
    // And the strip is what places them: the pill at one end, the link at the
    // other. Without this the cancelled margin would leave the two adjacent.
    expect(styleOf(RAIL, ".pg-rail-foot").justifyContent).toBe("space-between");
  });
});
