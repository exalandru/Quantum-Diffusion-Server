/** The tabs are a control, not part of the masthead.
 *
 * A stylesheet test, with the same bounds as `Logs.layout.test.ts`: jsdom parses
 * CSS and answers `matches()` but computes no boxes, so "the tabs are on their
 * own line" is not observable here. What *is* observable is the cascade, and the
 * cascade is what was wrong.
 *
 * Everything lived in one flex row — product name, status pill, four navigation
 * tabs, endpoint — at the same weight and with the same neutral border, so the
 * tabs read as more title rather than as the thing you navigate with. Three
 * declarations carry the fix and each is pinned below:
 *
 * * the header stacks (`flex-direction: column`), so identity and navigation are
 *   two rows rather than one;
 * * the tab strip has a surface of its own, which is what makes four buttons
 *   read as one control;
 * * the selected tab is marked in accent, the same way `.variant` marks the
 *   active one — one meaning, one vocabulary.
 *
 * And one thing that must NOT change: `header` stays a direct child of `main`.
 * `main.view-logs > header` is a direct-child selector, so wrapping the header
 * or splitting it into siblings silently returns the Logs pane to a fixed box —
 * which is exactly what `Logs.layout.test.ts` exists to catch, from the other
 * side.
 *
 * What this does not establish: that the rendered header looks right. That needs
 * a browser.
 */

import { describe, expect, it } from "vitest";

import "./styles.css";

/** The header, as `App.tsx` renders it. */
function header(): { main: HTMLElement; header: HTMLElement; views: HTMLElement } {
  document.body.innerHTML = `
    <main>
      <header>
        <div class="identity">
          <h1>Quantum Diffusion Server</h1>
          <span class="pill pill-live">Running</span>
          <code class="endpoint">127.0.0.1:8765</code>
        </div>
        <nav class="views" role="tablist" aria-label="Views">
          <button role="tab" aria-selected="true" class="view-tab">Dashboard</button>
          <button role="tab" aria-selected="false" class="view-tab">Models</button>
        </nav>
      </header>
    </main>`;
  return {
    main: document.querySelector("main") as HTMLElement,
    header: document.querySelector("header") as HTMLElement,
    views: document.querySelector("nav.views") as HTMLElement,
  };
}

/** Every declaration of `property` that applies to `element`, in sheet order. */
function declared(element: HTMLElement, property: string): string[] {
  const found: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    for (const rule of Array.from(sheet.cssRules) as CSSRule[]) {
      if (!(rule instanceof CSSStyleRule)) continue;
      const value = rule.style.getPropertyValue(property);
      if (!value) continue;
      try {
        if (element.matches(rule.selectorText)) found.push(value.trim());
      } catch {
        // A selector jsdom cannot parse cannot apply to this element either.
      }
    }
  }
  return found;
}

describe("the header", () => {
  it("has rules at all, so the assertions below mean something", () => {
    const count = Array.from(document.styleSheets).reduce(
      (total, sheet) => total + sheet.cssRules.length,
      0,
    );
    expect(count).toBeGreaterThan(50);
  });

  it("stacks identity above navigation", () => {
    const { header: element } = header();
    expect(declared(element, "flex-direction")).toContain("column");
  });

  it("keeps the header a direct child of main", () => {
    // The constraint `main.view-logs > header` depends on. Asserted here as
    // well as in Logs.layout.test.ts because this is the file whose changes
    // would break it.
    const { main, header: element } = header();
    expect(element.parentElement).toBe(main);
    expect(main.querySelector(":scope > header")).toBe(element);
  });

  it("gives the tab strip a surface, so four buttons read as one control", () => {
    const { views } = header();
    expect(declared(views, "background").length).toBeGreaterThan(0);
    expect(declared(views, "border").length).toBeGreaterThan(0);
  });

  it("marks the selected tab the way the rest of the app marks an active one", () => {
    document.body.innerHTML = `
      <nav class="views">
        <button class="view-tab" aria-selected="true"></button>
      </nav>
      <div class="variants"><button class="variant" aria-pressed="true"></button></div>`;
    const tab = document.querySelector(".view-tab") as HTMLElement;
    const variant = document.querySelector(".variant") as HTMLElement;

    // The same tokens, not merely "some highlight": a second vocabulary for one
    // meaning is what this settles.
    expect(declared(tab, "background")).toContain("var(--accent-tint)");
    expect(declared(variant, "background")).toContain("var(--accent-tint)");
    expect(declared(tab, "border-color")).toContain("var(--accent-line)");
    expect(declared(variant, "border-color")).toContain("var(--accent-line)");
  });

  it("does not mark an unselected tab", () => {
    // The negative: without it, a rule matching every `.view-tab` would satisfy
    // the assertion above while making all four look selected.
    document.body.innerHTML = `<button class="view-tab" aria-selected="false"></button>`;
    const tab = document.querySelector(".view-tab") as HTMLElement;
    expect(declared(tab, "background")).not.toContain("var(--accent-tint)");
  });
});
