/** A component row reads left to right: box, name, note — then status, hard right.
 *
 * A stylesheet test, with the same bounds as `Logs.layout.test.ts`: jsdom parses
 * CSS and answers `matches()` but computes no boxes, so "the pill is aligned with
 * the one above it" is not observable here. What *is* observable is the cascade,
 * and the cascade is what was wrong.
 *
 * The row was `[checkbox + name]` as a `flex: 1` label between a left edge and a
 * right-hand pill. That centres nothing on purpose and everything in practice:
 * the label took the whole middle, its contents sat wherever its own alignment
 * put them, and "Text encoder" broke over two lines with empty space at both
 * ends of the row it was supposedly filling.
 *
 * So three declarations carry the fix, and each is pinned below:
 *
 * * the identity group starts at the left (`justify-content: flex-start`);
 * * the status is pushed to the right by the free space (`margin-left: auto`),
 *   which is what makes pills line up down the list whatever the names measure;
 * * the name does not wrap (`white-space: nowrap`) while the row still may, so a
 *   narrow window wraps between the parts rather than inside a word.
 *
 * What this does not establish: that the rendered row looks right. That needs a
 * browser, and it belongs to the packaged visual witnesses.
 */

import { describe, expect, it } from "vitest";

import "./styles.css";

/** One component row, as `Models.tsx` renders it inside the dialog. */
function componentRow(): { entry: HTMLElement; label: HTMLElement; pill: HTMLElement } {
  document.body.innerHTML = `
    <ul class="components">
      <li class="component">
        <label class="check component-id">
          <input type="checkbox" />
          <span class="component-name">Text encoder</span>
          <span class="component-note">saved at source precision</span>
        </label>
        <span class="pill">Not converted</span>
      </li>
    </ul>`;
  const entry = document.querySelector("li.component") as HTMLElement;
  const label = document.querySelector("label.component-id") as HTMLElement;
  const pill = document.querySelector("li.component > span.pill") as HTMLElement;
  if (!entry || !label || !pill) throw new Error("the component row changed shape");
  return { entry, label, pill };
}

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

function declared(element: HTMLElement, property: string): string[] {
  return styleRules()
    .filter(({ selector }) => {
      try {
        return element.matches(selector);
      } catch {
        return false;
      }
    })
    .map(({ style }) => style.getPropertyValue(property))
    .filter((value) => value !== "");
}

describe("component rows align on their identity, not on their middle", () => {
  it("starts the checkbox and name at the left of the row", () => {
    const { label } = componentRow();
    expect(declared(label, "justify-content")).toContain("flex-start");
    // It still spans the row: it is the click target for the whole identity.
    expect(declared(label, "flex").join(" ")).toContain("1 1 auto");
  });

  it("pushes the status pill to the right with the free space", () => {
    const { pill } = componentRow();
    expect(declared(pill, "margin-left")).toContain("auto");
    // And never lets it shrink, so the column it forms stays a column.
    expect(declared(pill, "flex").join(" ")).toContain("0 0 auto");
  });

  it("keeps the component name on one line while the row may still wrap", () => {
    const { entry, label } = componentRow();
    expect(declared(document.querySelector(".component-name") as HTMLElement, "white-space")).toContain(
      "nowrap",
    );
    // Deliberate wrapping at narrow widths — between the parts, not inside them.
    expect(declared(entry, "flex-wrap")).toContain("wrap");
    expect(declared(label, "flex-wrap")).toContain("wrap");
  });

  it("would notice a stubbed stylesheet", () => {
    expect(styleRules().length).toBeGreaterThan(20);
  });
});
