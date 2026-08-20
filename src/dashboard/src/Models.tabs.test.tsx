/**
 * The catalogue is two tabs, and only one of them is on screen.
 *
 * Open and gated built-ins used to be stacked sections, which put five
 * repositories the reader may have no access to between them and the ones they
 * can install. The split itself is still the backend's `gated` field — what
 * changed is that only the selected half is rendered, so this pins both halves of
 * that: the selected list is there, and the other one is *not*.
 *
 * A separate file from `Models.test.tsx` on purpose: that suite still drives the
 * panel through the Tauri `invoke` bridge this app no longer has, so it cannot
 * run. Here the seam is `fetch`, which is what the panel actually uses.
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
  model({ key: "flux2-klein", display_name: "FLUX.2-klein", gated: true }),
  model({ key: "z-image", display_name: "Z-Image" }),
  model({ key: "fibo", display_name: "FIBO", gated: true }),
  model({ key: "z-image-turbo", display_name: "Z-Image Turbo" }),
  model({
    key: "local-abc",
    display_name: "My local model",
    provenance: "imported_local",
    can_download: false,
  }),
];

/** The model names inside one list, in document order. */
function namesIn(list: HTMLElement): string[] {
  return within(list)
    .getAllByRole("heading")
    .map((heading) => heading.textContent ?? "");
}

it("opens on the models that need no account, and hides the gated half", async () => {
  show(CATALOGUE);

  const open = await screen.findByRole("list", { name: "Open models" });
  expect(namesIn(open)).toEqual(["Z-Image", "Z-Image Turbo"]);
  expect(screen.queryByRole("list", { name: "Gated models" })).toBeNull();

  const tabs = screen.getByRole("tablist", { name: "Catalogue" });
  const [openTab, gatedTab] = within(tabs).getAllByRole("tab");
  expect(openTab.getAttribute("aria-selected")).toBe("true");
  expect(gatedTab.getAttribute("aria-selected")).toBe("false");
  // Each tab counts its own half, whichever one is selected.
  expect(openTab.textContent).toBe("Open2");
  expect(gatedTab.textContent).toBe("Gated2");
});

it("swaps which half is rendered when the other tab is chosen", async () => {
  show(CATALOGUE);
  const tabs = await screen.findByRole("tablist", { name: "Catalogue" });
  const [openTab, gatedTab] = within(tabs).getAllByRole("tab");

  await userEvent.click(gatedTab);

  expect(namesIn(screen.getByRole("list", { name: "Gated models" }))).toEqual([
    "FLUX.2-klein",
    "FIBO",
  ]);
  expect(screen.queryByRole("list", { name: "Open models" })).toBeNull();
  expect(gatedTab.getAttribute("aria-selected")).toBe("true");
  expect(openTab.getAttribute("aria-selected")).toBe("false");

  await userEvent.click(openTab);
  expect(screen.getByRole("list", { name: "Open models" })).toBeTruthy();
  expect(screen.queryByRole("list", { name: "Gated models" })).toBeNull();
});

it("keeps imported local models out of both tabs", async () => {
  show(CATALOGUE);
  await screen.findByRole("list", { name: "Open models" });

  // Their own panel, reachable from either tab: provenance is a different
  // question from gating, and it was never one of these two halves.
  const imported = screen.getByRole("list", { name: "Imported local models" });
  expect(namesIn(imported)).toEqual(["My local model"]);

  await userEvent.click(within(screen.getByRole("tablist", { name: "Catalogue" })).getAllByRole("tab")[1]);
  expect(namesIn(screen.getByRole("list", { name: "Imported local models" }))).toEqual([
    "My local model",
  ]);
});

it("says a half is empty rather than rendering nothing", async () => {
  // A tab whose panel draws nothing at all reads as a broken view.
  show([model({ key: "z-image", display_name: "Z-Image" })]);

  const tabs = await screen.findByRole("tablist", { name: "Catalogue" });
  const gatedTab = within(tabs).getAllByRole("tab")[1];
  expect(gatedTab.textContent).toBe("Gated0");

  await userEvent.click(gatedTab);

  expect(screen.getByText("Nothing in this half of the catalogue.")).toBeTruthy();
  expect(screen.queryByRole("list", { name: "Gated models" })).toBeNull();
});

it("points the missing-token caution at the tab the affected rows are in", async () => {
  // The caution sits above the tablist, so it is usually read while the *open*
  // half is on screen: "these repositories" named rows that were not there.
  show(CATALOGUE, overview({ hfTokenPresent: false }));
  await screen.findByRole("list", { name: "Open models" });

  const caution = screen.getByText(/will refuse a download with a 401/);
  expect(caution.textContent).toContain("2 repositories in the");
  expect(within(caution).getByText("Gated")).toBeTruthy();
});
