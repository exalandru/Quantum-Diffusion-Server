/**
 * The catalogue is read by release, not by model name.
 *
 * Seventeen names in a flat list are read one at a time; nine releases are read
 * by release, and the choice inside one — Turbo or Aesthetic, Medium or Large —
 * is the one the reader actually faces. So each release is a fieldset, the same
 * one the settings form groups by.
 *
 * Both facts under test are the backend's and neither is recomputed here: which
 * release a row belongs to is its `group_label`, and which model to reach for
 * first is the order the catalogue publishes. This pins that the panel *renders*
 * both faithfully — a component that sorted, deduped or regrouped on its own
 * would fail here.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { Models } from "./panels/Models";
import type { JobView } from "./job";
import { catalogue, job, model, overview } from "./test-fixtures";
import type { ModelStatus } from "./types";

function idleJobs(): JobView {
  return {
    job: job(),
    error: null,
    active: false,
    refresh: vi.fn(async () => {}),
    onSettled: () => () => {},
  };
}

function show(models: ModelStatus[]) {
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
  return render(
    <Models
      state={overview()}
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

/** The model names inside one list, in document order. */
function namesIn(list: HTMLElement): string[] {
  return within(list)
    .getAllByRole("heading")
    .map((heading) => heading.textContent ?? "");
}

/**
 * Every model name in one half of the catalogue, in document order, across all
 * of its sections.
 *
 * Reads the rows rather than every heading in the subtree: the half carries its
 * own heading, and each release carries a legend, neither of which is a model.
 */
function namesAcross(half: HTMLElement): string[] {
  return within(half)
    .getAllByRole("listitem")
    .map((row) => within(row).getAllByRole("heading")[0]?.textContent ?? "");
}

/** One model's row, so a query cannot accidentally match its neighbour. */
async function rowFor(name: string): Promise<HTMLElement> {
  const heading = await screen.findByText(name);
  const row = heading.closest("li");
  if (!row) throw new Error(`no row for ${name}`);
  return row;
}

/**
 * Two releases in the open half, in the order the backend publishes them:
 * lightest first inside each.
 *
 * Deliberately not the real catalogue — a fixture that mirrors the backend
 * invites being read as the backend. What matters is the shape: two releases,
 * one of them holding more than one model.
 */
const OPEN = [
  model({ key: "anima-turbo", display_name: "Anima Turbo", group_label: "Anima" }),
  model({ key: "anima", display_name: "Anima Aesthetic", group_label: "Anima" }),
  model({ key: "z-image-turbo", display_name: "Z-Image Turbo", group_label: "Z-Image" }),
  model({ key: "z-image", display_name: "Z-Image", group_label: "Z-Image" }),
];

it("gathers a release into one fieldset, titled by the release", async () => {
  show(OPEN);

  const anima = await screen.findByRole("group", { name: "Anima" });
  expect(anima.tagName).toBe("FIELDSET");
  expect(namesIn(anima)).toEqual(["Anima Turbo", "Anima Aesthetic"]);

  const zImage = screen.getByRole("group", { name: "Z-Image" });
  expect(namesIn(zImage)).toEqual(["Z-Image Turbo", "Z-Image"]);
});

it("keeps the backend's order, which puts the lightest model of a release first", async () => {
  // The panel is not the authority on which model to reach for first, and it
  // must not become one: a component that sorted by name would put "Anima
  // Aesthetic" above "Anima Turbo" and read as the opposite recommendation.
  show(OPEN);

  const half = await screen.findByRole("region", { name: "Open models" });
  expect(namesAcross(half)).toEqual(["Anima Turbo", "Anima Aesthetic", "Z-Image Turbo", "Z-Image"]);
});

it("boxes a release that ships a single model like any other", async () => {
  // Krea 2 is a named release with one model in it. Left unboxed it would read
  // as a differently-shaped thing rather than as a release of one.
  show([model({ key: "krea-2-turbo", display_name: "Krea 2 Turbo", group_label: "Krea 2" })]);

  const krea = await screen.findByRole("group", { name: "Krea 2" });
  expect(namesIn(krea)).toEqual(["Krea 2 Turbo"]);
});

it("groups by the release, not by the architecture family", async () => {
  // FLUX.2 klein and dev ship as one release and load as two families. `family`
  // decides what the engine does; it must not decide how the list reads.
  show([
    model({
      key: "flux2-klein",
      display_name: "Flux 2 Klein",
      family: "flux2",
      group_label: "FLUX.2",
      gated: true,
    }),
    model({
      key: "flux2-dev",
      display_name: "Flux 2 Dev",
      family: "flux2-dev",
      group_label: "FLUX.2",
      gated: true,
    }),
  ]);

  const tablist = await screen.findByRole("tablist", { name: "Catalogue" });
  await userEvent.click(within(tablist).getAllByRole("tab")[1]!);

  const flux = await screen.findByRole("group", { name: "FLUX.2" });
  expect(namesIn(flux)).toEqual(["Flux 2 Klein", "Flux 2 Dev"]);
});

it("renders a release the backend split into two runs as two sections", async () => {
  // Not stitched back together: contiguity is the backend's invariant, pinned by
  // `test_each_release_is_one_contiguous_run_of_the_catalogue`. If it ever
  // breaks, the honest rendering is what the backend actually published — a
  // panel that silently regrouped would hide the bug rather than show it.
  show([
    model({ key: "anima-turbo", display_name: "Anima Turbo", group_label: "Anima" }),
    model({ key: "z-image-turbo", display_name: "Z-Image Turbo", group_label: "Z-Image" }),
    model({ key: "anima", display_name: "Anima Aesthetic", group_label: "Anima" }),
  ]);

  await screen.findByRole("region", { name: "Open models" });
  const sections = screen.getAllByRole("group", { name: "Anima" });
  expect(sections.map(namesIn)).toEqual([["Anima Turbo"], ["Anima Aesthetic"]]);
});

it("lists rows a backend declares no release for without inventing one", async () => {
  // `group_label: null` is "this backend does not say", which is not the same as
  // a release of one: a box legended with the model's own name would assert a
  // grouping nobody published.
  show([model({ key: "z-image", display_name: "Z-Image", group_label: null })]);

  const half = await screen.findByRole("region", { name: "Open models" });
  expect(namesAcross(half)).toEqual(["Z-Image"]);
  expect(within(half).queryByRole("group")).toBeNull();
});

it("says the architecture family only where no legend already says it", async () => {
  // Two rows, same family, different surroundings. A catalogue row sits under a
  // legend naming its release, so repeating `z-image` beneath it is the same
  // fact twice in worse words. An imported row has no legend above it, and
  // there the detected family is what says what the directory turned out to
  // hold — so it stays.
  show([
    model({
      key: "z-image-turbo",
      display_name: "Z-Image Turbo",
      repo: "mlx-community/Z-Image-Turbo-bf16",
      family: "z-image",
      group_label: "Z-Image",
    }),
    model({
      key: "local-abc",
      display_name: "My local Z-Image",
      repo: "/models/mine",
      family: "z-image",
      group_label: null,
      provenance: "imported_local",
      can_download: false,
    }),
  ]);

  const catalogueRow = within(await rowFor("Z-Image Turbo"));
  expect(catalogueRow.queryByText("z-image")).toBeNull();

  const importedRow = within(await rowFor("My local Z-Image"));
  expect(importedRow.getByText("z-image")).toBeTruthy();
});
