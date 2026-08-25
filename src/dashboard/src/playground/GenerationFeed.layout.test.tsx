/**
 * Why a feed row of pictures is level, and why its toolbars are reachable.
 *
 * The reported bug: the cell carrying "Upscaled · 2880x1600" was taller than
 * its neighbours, so its action buttons sat lower and the row looked broken.
 * The cause was structural — the `<figcaption>` and the toolbar were siblings
 * *below* `.pg-thumb`, in normal flow, so a cell with a caption was taller than
 * a cell without one, and the grid row grew to the tallest cell.
 *
 * jsdom has no layout engine, so heights cannot be measured here. What can be
 * asked, of the real cascade (`css: true`), is the property that *caused* the
 * unequal heights: how many of a cell's children contribute height at all. An
 * upscale cell and a plain cell now contribute the same one — the picture — and
 * that is asserted as a comparison between the two cells rather than as a rule
 * quoted back, so a change that took only the caption out of flow and left the
 * toolbar in would still fail.
 *
 * The real components are rendered rather than a markup fixture because the
 * defect was the markup: the sheet's `.pg-tile` rules had been doing this
 * correctly in the gallery all along.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import "../styles.css";
import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { GenerationFeed } from "./GenerationFeed";

const IDLE: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  preview_seq: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

const UPSCALERS: Upscaler[] = [
  {
    id: "realesrgan-x4plus",
    name: "Real-ESRGAN ×4 (photo)",
    scales: [2, 4],
    downloaded: true,
    sizeMb: 33.5,
    license: "BSD-3-Clause",
  },
];

// jsdom has no layout engine and so no `scrollIntoView`; the feed calls it to
// keep new activity in view, which is not what this file observes.
Element.prototype.scrollIntoView = () => {};

function generation(patch: Partial<PlaygroundGeneration> = {}): PlaygroundGeneration {
  const id = patch.id ?? "g1";
  return {
    id,
    sessionId: "s1",
    groupId: id,
    prompt: "a fox",
    rewrittenPrompt: null,
    rewriteError: null,
    negativePrompt: null,
    model: "qwen-image-2512",
    kind: "txt2img",
    n: 1,
    size: "1440x800",
    steps: 28,
    seeds: [41],
    contextImage: null,
    status: "completed",
    error: null,
    images: [{ url: `/playground/images/${id}.png`, seed: 41 }],
    createdAt: 1,
    startedAt: 1,
    finishedAt: 2,
    ...patch,
  };
}

/**
 * One entry holding two pictures: an ordinary render, and an upscale of it.
 *
 * The upscale joins the root's group, which is what puts the two in one grid
 * row — the exact arrangement the screenshot showed broken.
 */
function rowWithAnUpscale() {
  const root = generation({ id: "root" });
  const upscale = generation({
    id: "up",
    groupId: "root",
    kind: "upscale",
    size: "2880x1600",
    steps: 0,
    createdAt: 2,
  });
  return render(
    <GenerationFeed
      generations={[root, upscale]}
      progress={IDLE}
      onCancel={vi.fn()}
      cancelling={null}
      busy={false}
      onRefine={vi.fn()}
      onVariation={vi.fn()}
      onUpscale={vi.fn()}
      upscalers={UPSCALERS}
      onDeleteImage={vi.fn()}
      onDeleteGroup={vi.fn()}
      paused={false}
      nameOf={(id) => id}
    />,
  );
}

/**
 * The children of a cell that still take part in normal flow, by class.
 *
 * Stated as "not taken out of flow" rather than `position === "static"`: jsdom
 * reports the empty string for a property no rule sets, so asking for the
 * initial value by name would call every untouched element out-of-flow and the
 * comparison below would pass on any markup at all.
 */
function inFlow(cell: Element): string[] {
  const OUT = ["absolute", "fixed"];
  return [...cell.children]
    .filter((child) => !OUT.includes(window.getComputedStyle(child).position))
    .map((child) => child.className);
}

it("gives every cell of a row the same height, upscale or not", () => {
  const { container } = rowWithAnUpscale();
  const cells = [...container.querySelectorAll(".pg-image-cell")];
  expect(cells).toHaveLength(2);

  // The upscale is the cell that carries a caption; the other carries none.
  const tagged = cells.filter((cell) => cell.querySelector(".pg-image-tag") !== null);
  expect(tagged).toHaveLength(1);

  // …and it contributes exactly the same height as its neighbour, because the
  // caption is not in flow. This is the comparison, not a quoted rule: it fails
  // if either the caption or the toolbar goes back into the cell's flow.
  const [first, second] = cells as [Element, Element];
  expect(inFlow(first)).toEqual(["pg-thumb"]);
  expect(inFlow(second)).toEqual(["pg-thumb"]);

  // The cell is what the overlays resolve against.
  expect(window.getComputedStyle(first).position).toBe("relative");
});

it("hides the tools at rest and reveals them to the keyboard", async () => {
  const { container } = rowWithAnUpscale();
  const actions = container.querySelector(".pg-image-cell .pg-image-actions") as HTMLElement;

  // At rest: invisible, but still in the tab order. `visibility: hidden` or
  // `display: none` would take the buttons out of it, so focus could never
  // arrive to trigger the reveal — the reveal rule would be self-defeating.
  expect(window.getComputedStyle(actions).opacity).toBe("0");
  expect(window.getComputedStyle(actions).visibility).not.toBe("hidden");
  expect(window.getComputedStyle(actions).display).not.toBe("none");

  // Reachable by tabbing, not only by hovering. jsdom does not evaluate
  // `:focus-within` when it computes a style, so the reveal itself is asserted
  // against the sheet; what a render can prove is that the focus the rule waits
  // for can actually land, which is the half that was self-defeating before.
  const refine = screen.getAllByRole("button", { name: /Refine/ })[0] as HTMLElement;
  await userEvent.tab();
  expect(actions.contains(document.activeElement)).toBe(false);
  refine.focus();
  expect(actions.contains(document.activeElement)).toBe(true);

  const selectors = [...document.styleSheets].flatMap((sheet) =>
    [...sheet.cssRules].flatMap((rule) =>
      rule instanceof CSSStyleRule ? rule.selectorText.split(",").map((one) => one.trim()) : [],
    ),
  );
  expect(selectors).toContain(".pg-image-cell .pg-image-actions:focus-within");
  expect(selectors).toContain(".pg-image-cell:hover .pg-image-actions");
});

it("draws the upscale tag over the picture, and keeps it readable at rest", () => {
  const { container } = rowWithAnUpscale();
  const tag = container.querySelector(".pg-image-tag") as HTMLElement;
  const style = window.getComputedStyle(tag);
  expect(style.position).toBe("absolute");
  // Not revealed on hover like the toolbar: which of two otherwise identical
  // pictures is the 2880x1600 one is a fact, not an action.
  expect(style.opacity).not.toBe("0");
  // And it cannot swallow a click meant for the picture underneath it.
  expect(style.pointerEvents).toBe("none");
});
