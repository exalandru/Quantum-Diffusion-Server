/**
 * Why the history sidebar's rows are a name and not a toolbar.
 *
 * Both of these are cascade questions, so they are asked of the real cascade
 * (`css: true` in the vitest config) rather than assumed. jsdom has no layout
 * engine, so widths cannot be measured here — but the two properties that
 * *caused* the collapse are computed styles, and they are what these pin.
 *
 * What went wrong: the action buttons sat in the row's flex line with
 * `flex-shrink: 0`, so they took their full intrinsic width — about 330px of
 * text buttons — out of a 280px sidebar and never gave any back. The name, with
 * `min-width: 0`, absorbed the whole shortfall and collapsed to a few letters.
 * And because they were hidden with `visibility: hidden`, which hides an element
 * without freeing the space it occupies, that happened on every row in the list,
 * not only the one under the pointer.
 */

import { describe, expect, it } from "vitest";

import "../styles.css";

const ROW = `
  <aside class="pg-sidebar">
    <ul class="pg-sessions">
      <li>
        <div class="pg-session-row">
          <button class="pg-session-open">
            <span class="pg-session-title">At night, a fox crosses the road</span>
            <span class="pg-session-meta">just now</span>
          </button>
          <div class="pg-session-actions" role="group">
            <button class="small pg-tool" title="Rename"></button>
            <button class="small danger pg-tool" title="Delete"></button>
          </div>
        </div>
      </li>
    </ul>
  </aside>`;

/** Every selector in the loaded sheet, for a rule jsdom will not compute. */
function rules(): string[] {
  return [...document.styleSheets].flatMap((sheet) =>
    [...sheet.cssRules].map((rule) => (rule as CSSStyleRule).selectorText ?? ""),
  );
}

function styleOf(selector: string): CSSStyleDeclaration {
  document.body.innerHTML = ROW;
  const element = document.querySelector(selector);
  if (!element) throw new Error(`no ${selector}`);
  return window.getComputedStyle(element);
}

describe("a session row", () => {
  it("takes its actions out of flow, so they cost the name no width", () => {
    // The whole fix. In flow, these buttons were the reason a title read "At ni…".
    expect(styleOf(".pg-session-actions").position).toBe("absolute");
    expect(styleOf(".pg-session-row").position).toBe("relative");
  });

  it("hides them with opacity, so keyboard focus can still reveal them", () => {
    // Not `visibility: hidden`: that removes a button from the tab order, so it
    // could never take the focus the reveal rule was waiting for. The rule was
    // self-defeating, and the comment above it claimed otherwise.
    const actions = styleOf(".pg-session-actions");
    expect(actions.opacity).toBe("0");
    expect(actions.visibility).not.toBe("hidden");
    expect(actions.pointerEvents).toBe("none");
  });

  it("still lets the name itself elide when it is too long", () => {
    const title = styleOf(".pg-session-title");
    expect(title.overflow).toBe("hidden");
    expect(title.textOverflow).toBe("ellipsis");
    expect(title.whiteSpace).toBe("nowrap");
  });

  it("paints no empty tooltip box for a tool using the browser's own", () => {
    // `.pg-tool::after` is `content: attr(data-tip)`, which resolves to an empty
    // string when the attribute is absent — a bordered box around nothing. These
    // sidebar tools deliberately carry no `data-tip`, because the sidebar
    // scrolls and would clip a tooltip drawn outside the button.
    document.body.innerHTML = ROW;
    expect(document.querySelector(".pg-tool")?.hasAttribute("data-tip")).toBe(false);

    // Asserted against the sheet rather than the computed style: jsdom does not
    // resolve pseudo-element styles, so this proves the guard is *present*, not
    // that a browser applies it. The narrower claim is the true one.
    expect(rules()).toContain(".pg-tool:not([data-tip])::after");
  });
});
