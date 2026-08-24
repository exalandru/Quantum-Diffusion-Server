/**
 * The catalogue is two tabs, and only one of them is on screen.
 *
 * Open and gated built-ins used to be stacked sections, which put five
 * repositories the reader may have no access to between them and the ones they
 * can install. The split itself is still the backend's `gated` field — what
 * changed is that only the selected half is rendered, so this pins both halves of
 * that: the selected list is there, and the other one is *not*.
 *
 * A separate file from `Models.test.tsx` on purpose: this one is about the tabs
 * themselves — which half is rendered and which is not — rather than about what
 * a row may offer. Both drive the panel over `fetch`, which is what it uses.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { Models } from "./panels/Models";
import type { JobView } from "./job";
import { catalogue, job, model, overview } from "./test-fixtures";
import type { ModelStatus, Overview } from "./types";

function idleJobs(): JobView {
  return {
    job: job(),
    error: null,
    active: false,
    refresh: vi.fn(async () => {}),
    onSettled: () => () => {},
  };
}

/**
 * The two GETs the panel makes on mount.
 *
 * `/v1/capabilities` answers 503 because the server being stopped is the normal
 * case for this screen — it is when you most want to download weights — and the
 * panel must render the catalogue anyway.
 */
function serve(models: ModelStatus[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(typeof input === "string" ? input : input instanceof URL ? input : "");
      if (url.includes("/admin/models")) {
        return new Response(JSON.stringify(catalogue(models)), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.includes("/capabilities")) {
        return new Response(JSON.stringify({ error: { message: "not running" } }), {
          status: 503,
          headers: { "content-type": "application/json" },
        });
      }
      throw new Error(`unexpected request ${url}`);
    }),
  );
}

function show(models: ModelStatus[], state: Overview = overview()) {
  serve(models);
  return render(
    <Models
      state={state}
      config={{}}
      jobs={idleJobs()}
      onConfigChanged={vi.fn(async () => {})}
    />,
  );
}

beforeEach(() => {
  vi.stubGlobal("scrollTo", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const CATALOGUE = [
  model({
    key: "flux2-klein",
    display_name: "FLUX.2-klein",
    gated: true,
    group_label: "FLUX.2",
  }),
  model({ key: "z-image", display_name: "Z-Image", group_label: "Z-Image" }),
  model({ key: "fibo", display_name: "FIBO", gated: true, group_label: "FIBO" }),
  model({ key: "z-image-turbo", display_name: "Z-Image Turbo", group_label: "Z-Image" }),
  model({
    key: "local-abc",
    display_name: "My local model",
    provenance: "imported_local",
    can_download: false,
  }),
];

/**
 * The catalogue's two tabs, in order.
 *
 * Two is the component's invariant rather than this test's convenience:
 * `CATALOGUE_TABS` in `Models.tsx` is a two-row table and the tablist renders
 * exactly one control per row. Asserting the length is what makes indexing into
 * the result safe instead of assumed — if a third half ever appears, this fails
 * here and says so, rather than silently reading `undefined`.
 */
async function catalogueTabs(): Promise<[HTMLElement, HTMLElement]> {
  const tablist = await screen.findByRole("tablist", { name: "Catalogue" });
  const tabs = within(tablist).getAllByRole("tab");
  expect(tabs).toHaveLength(2);
  return [tabs[0]!, tabs[1]!];
}

/** The model names inside one list, in document order. */
function namesIn(list: HTMLElement): string[] {
  return within(list)
    .getAllByRole("heading")
    .map((heading) => heading.textContent ?? "");
}

/**
 * One half of the catalogue, addressed the way the tab names it.
 *
 * The half is the region, not a list: it holds one list per release now, and
 * none of them is named after the tab. Querying a list called "Open models"
 * would find nothing whether or not the half is on screen, which is a negative
 * assertion that can no longer fail — the region is what actually appears and
 * disappears with the selected tab.
 */
function half(name: "Open models" | "Gated models"): HTMLElement | null {
  return screen.queryByRole("region", { name });
}

/** Every model name in one half, across all of its releases, in document order. */
function namesInHalf(name: "Open models" | "Gated models"): string[] {
  const region = half(name);
  if (!region) throw new Error(`the ${name} half is not on screen`);
  return within(region)
    .getAllByRole("listitem")
    .map((row) => within(row).getAllByRole("heading")[0]?.textContent ?? "");
}

it("opens on the models that need no account, and hides the gated half", async () => {
  show(CATALOGUE);

  await screen.findByRole("region", { name: "Open models" });
  expect(namesInHalf("Open models")).toEqual(["Z-Image", "Z-Image Turbo"]);
  expect(half("Gated models")).toBeNull();

  const [openTab, gatedTab] = await catalogueTabs();
  expect(openTab.getAttribute("aria-selected")).toBe("true");
  expect(gatedTab.getAttribute("aria-selected")).toBe("false");
  // Each tab counts its own half, whichever one is selected.
  expect(openTab.textContent).toBe("Open2");
  expect(gatedTab.textContent).toBe("Gated2");
});

it("swaps which half is rendered when the other tab is chosen", async () => {
  show(CATALOGUE);
  const [openTab, gatedTab] = await catalogueTabs();

  await userEvent.click(gatedTab);

  expect(namesInHalf("Gated models")).toEqual(["FLUX.2-klein", "FIBO"]);
  expect(half("Open models")).toBeNull();
  expect(gatedTab.getAttribute("aria-selected")).toBe("true");
  expect(openTab.getAttribute("aria-selected")).toBe("false");

  await userEvent.click(openTab);
  expect(half("Open models")).toBeTruthy();
  expect(half("Gated models")).toBeNull();
});

it("keeps imported local models out of both tabs", async () => {
  show(CATALOGUE);
  await screen.findByRole("region", { name: "Open models" });

  // Their own panel, reachable from either tab: provenance is a different
  // question from gating, and it was never one of these two halves.
  const imported = screen.getByRole("list", { name: "Imported local models" });
  expect(namesIn(imported)).toEqual(["My local model"]);

  const [, gated] = await catalogueTabs();
  await userEvent.click(gated);
  expect(namesIn(screen.getByRole("list", { name: "Imported local models" }))).toEqual([
    "My local model",
  ]);
});

it("says a half is empty rather than rendering nothing", async () => {
  // A tab whose panel draws nothing at all reads as a broken view.
  show([model({ key: "z-image", display_name: "Z-Image" })]);

  const [, gatedTab] = await catalogueTabs();
  expect(gatedTab.textContent).toBe("Gated0");

  await userEvent.click(gatedTab);

  expect(screen.getByText("Nothing in this half of the catalogue.")).toBeTruthy();
  // The half itself is on screen and holds no list at all — which is the point:
  // an empty tab that rendered nothing would read as a broken view.
  const empty = half("Gated models");
  expect(empty).toBeTruthy();
  expect(within(empty!).queryAllByRole("list")).toEqual([]);
});

it("points the missing-token caution at the tab the affected rows are in", async () => {
  // The caution sits above the tablist, so it is usually read while the *open*
  // half is on screen: "these repositories" named rows that were not there.
  show(CATALOGUE, overview({ hfTokenPresent: false }));
  await screen.findByRole("region", { name: "Open models" });

  const caution = screen.getByText(/will refuse a download with a 401/);
  expect(caution.textContent).toContain("2 repositories in the");
  expect(within(caution).getByText("Gated")).toBeTruthy();
});
