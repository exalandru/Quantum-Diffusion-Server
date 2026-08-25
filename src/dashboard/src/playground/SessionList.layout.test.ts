/**
 * Why the rail's rows are a name and not a toolbar, and why the collapsed rail
 * is 56px of *track* rather than 56px of content inside a 280px column.
 *
 * All of these are cascade questions, so they are asked of the real cascade
 * (`css: true` in the vitest config) rather than assumed. jsdom has no layout
 * engine, so widths cannot be measured here — but the properties that *caused*
 * the collapse are computed styles, and they are what these pin.
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
    <div class="pg-sidebar-head">
      <h2>Projects</h2>
      <button class="small pg-rail-toggle"></button>
      <button class="primary small pg-rail-new">+</button>
    </div>
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

/** The same rail at its narrow width: landmarks, no names, no row actions. */
const COLLAPSED = `
  <aside class="pg-sidebar pg-sidebar-collapsed">
    <div class="pg-sidebar-head">
      <button class="small pg-rail-toggle"></button>
      <button class="primary small pg-rail-new">+</button>
    </div>
    <ul class="pg-sessions">
      <li><button class="pg-rail-mark" aria-current="true">F</button></li>
    </ul>
  </aside>`;

/** Every selector in the loaded sheet, for a rule jsdom will not compute. */
function rules(): string[] {
  return [...document.styleSheets].flatMap((sheet) =>
    [...sheet.cssRules].map((rule) => (rule as CSSStyleRule).selectorText ?? ""),
  );
}

function styleOf(selector: string, markup: string = ROW): CSSStyleDeclaration {
  document.body.innerHTML = markup;
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

describe("the collapsed rail", () => {
  it("narrows the flex track itself, not just what sits in it", () => {
    // `.pg-sidebar` is a flex item of `.pg-layout` with `flex: 0 0 280px`. A
    // collapse that set only `width` would leave that basis in force: the rail
    // would keep its 280px track and the landmarks would sit in a 56px column
    // inside it, with the studio no wider than before. So the basis is the
    // property under test, and the expanded rail is asserted alongside it —
    // otherwise a rule that narrowed *both* states would pass.
    expect(styleOf(".pg-sidebar", COLLAPSED).flexBasis).toBe("56px");
    expect(styleOf(".pg-sidebar").flexBasis).toBe("280px");
  });

  it("stacks the rail's own controls, which do not fit side by side at 56px", () => {
    expect(styleOf(".pg-sidebar-head", COLLAPSED).flexDirection).toBe("column");
    // Still a row when there is room for one: the heading and the two controls
    // share a line at full width.
    expect(styleOf(".pg-sidebar-head").flexDirection).not.toBe("column");
  });

  it("keeps a name that cannot elide from widening the rail", () => {
    // Nothing in this column is elidable — a landmark is one letter — so the
    // overflow has to be clipped rather than allowed to push the track open.
    expect(styleOf(".pg-sidebar", COLLAPSED).overflowX).toBe("hidden");
  });

  it("marks the open project on the landmark, which has no row to mark", () => {
    // Expanded, selection is `aria-selected` on `.pg-session-row` and the tint
    // comes from that. A landmark is a single button with no row around it, so
    // the selected state is drawn on the button and keyed off `aria-current`.
    expect(rules()).toContain('.pg-rail-mark[aria-current="true"]');
    expect(styleOf(".pg-rail-mark", COLLAPSED).position).toBe("relative");
  });
});
