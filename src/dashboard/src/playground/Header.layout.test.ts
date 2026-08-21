/**
 * The playground header's right-hand group.
 *
 * A cascade question, so it is asked of the cascade (`css: true` in the vitest
 * config) rather than assumed: both the Pause control and the way out of the
 * page carry an `auto` left margin, and two of those in one flex row split the
 * free space between them instead of grouping at the edge — which leaves the
 * button stranded in the middle of the header.
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
    <header>
      <div class="identity"><h1>Quantum Diffusion Server</h1></div>
      <button class="small pg-pause">Pause queue</button>
      <a class="shell-link" href="/dashboard/">Server Config</a>
    </header>
  </main>`;

describe("the playground header", () => {
  it("lays its controls out in a row, so nothing stretches", () => {
    // The shell's own header is a column; the playground's overrides it. A
    // button in a column flex row would span the whole header.
    expect(styleOf(HEADER, "main.playground > header").flexDirection).toBe("row");
  });

  it("pushes the pause control to the far edge", () => {
    expect(styleOf(HEADER, ".pg-pause").marginLeft).toBe("auto");
  });

  it("keeps the way out beside it rather than splitting the gap", () => {
    expect(styleOf(HEADER, ".shell-link").marginLeft).toBe("0px");
  });
});
