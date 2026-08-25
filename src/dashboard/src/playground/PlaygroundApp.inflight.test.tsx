/**
 * T8 — in every view, a submission the server accepted is visibly in flight
 * until it completes, fails or is cancelled.
 *
 * The defect these witness: the gallery and the light table were given no
 * `progress` and no cancel, on the grounds that they show pictures that exist.
 * A user who sent a prompt from either view got no sign the server had taken it
 * — the prompt "vanished into thin air", and was only found again by switching
 * to Prompts. Nothing was ever actually lost, because polling is driven by the
 * generations rather than by the view; what was missing was any way to know
 * that.
 *
 * Written through `PlaygroundApp` rather than against the two views directly,
 * for two reasons. The wiring *is* the defect — both views could always have
 * drawn a placeholder, and were simply never told anything — so a test that
 * handed `progress` to `GalleryView` itself would assert the one thing that was
 * never in doubt. And the progress snapshot arrives over the real SSE transport,
 * so the frames these push are the frames the server writes.
 *
 * The step preview is the same component the feed uses, addressing the same
 * one-slot route: `/playground/api/preview?v=<seq>` is asserted by URL here
 * precisely so "the gallery shows the live preview" cannot be satisfied by a
 * lookalike box that shows nothing.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { type FakeServer, fakeServer } from "../test-server";
import type { PlaygroundGeneration, Progress } from "../types";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;
/** What `GET /playground/api/sessions/s1` answers; a test rewrites it mid-run. */
let generations: PlaygroundGeneration[];
let paused: boolean;
/** Writes one SSE frame into the open progress stream. */
let push: ((chunk: Uint8Array) => void) | null;

const base = {
  sessionId: "s1",
  negativePrompt: null,
  rewrittenPrompt: null,
  rewriteError: null,
  model: "qwen-image-2512",
  kind: "txt2img" as const,
  steps: 6,
  contextImage: null,
  error: null,
  createdAt: 1,
  startedAt: 1,
  finishedAt: 2,
};

/** A picture the project already has, in a lineage of its own. */
const DONE: PlaygroundGeneration = {
  ...base,
  id: "g1",
  groupId: "g1",
  prompt: "a fox in the snow",
  n: 1,
  size: "512x288",
  seeds: [41],
  status: "completed",
  images: [{ url: "/playground/images/f1.png", seed: 41 }],
};

/**
 * The submission under test, in a second lineage, at a *different* aspect ratio.
 *
 * Landscape 16:9 against the first lineage's 16:9-but-smaller keeps the two
 * distinguishable while making the reserved-box assertion mean something: the
 * placeholder must reserve `1024 / 576`, which is this run's own shape and not
 * the project's first.
 */
const inFlight = (over: Partial<PlaygroundGeneration> = {}): PlaygroundGeneration => ({
  ...base,
  id: "g2",
  groupId: "g2",
  prompt: "a fox at dusk",
  n: 1,
  size: "1024x576",
  seeds: [42],
  status: "running",
  startedAt: 2,
  finishedAt: null,
  images: [],
  ...over,
});

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

const encoder = new TextEncoder();
/** One `/v1/progress` frame, as the server writes it. */
const frame = (over: Partial<Progress>): Uint8Array =>
  encoder.encode(`data: ${JSON.stringify({ ...IDLE, ...over })}\n\n`);

/** Denoising step `step` of six, with frame `step` in the server's preview slot. */
const generating = (step: number): Partial<Progress> => ({
  state: "generating",
  step,
  total: 6,
  preview_seq: step,
  elapsed_s: 3,
});

beforeEach(() => {
  paused = false;
  push = null;
  generations = [DONE, inFlight()];
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({ max_n: 4, default_model: "qwen-image-2512" }));
  server.on(
    "GET /v1/progress",
    () =>
      new Response(
        new ReadableStream({
          start(controller) {
            push = (chunk) => controller.enqueue(chunk);
          },
        }),
      ),
  );
  server.on("GET /playground/api/upscalers", () => ({ upscalers: [] }));
  server.on("GET /playground/api/sessions", () => ({
    paused,
    // `generating` is what the rail's live dot reads; the page's poll is driven
    // by the generations themselves, so this only has to be honest.
    sessions: [
      playgroundSession({
        id: "s1",
        title: "foxes",
        generating: generations.some(
          (entry) => entry.status === "queued" || entry.status === "running",
        ),
      }),
    ],
  }));
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations,
  }));
  window.history.replaceState(null, "", "/playground/?session=s1");
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
  vi.restoreAllMocks();
});

/**
 * Wait for the app to have subscribed, then write one frame.
 *
 * No flush beyond `act`'s own: the stream is read in a loop of awaits, so the
 * frame lands a turn or two later, and every caller of this is followed by a
 * `waitFor` that is happy to keep asking.
 */
async function progress(over: Partial<Progress>) {
  await waitFor(() => expect(push).not.toBeNull());
  await act(async () => void push?.(frame(over)));
}

async function open(user: UserEvent, tab: "Gallery" | "Light Table") {
  await screen.findByRole("tab", { name: tab });
  await user.click(screen.getByRole("tab", { name: tab }));
}

/** The gallery's cells, in document order, as "what each one is". */
const cells = (container: HTMLElement) =>
  // `.pg-gallery-row > *`, not `.pg-gallery-wall > *`: the wall holds rows and a
  // row holds cells, because the layout is justified rows (each row scaled to
  // span the width) rather than a grid of equal tracks. Flattened here so the
  // assertions stay "the wall's cells, in document order".
  [...container.querySelectorAll(".pg-gallery-row > *")].map((cell) =>
    cell.querySelector(".pg-tile-pending") !== null ||
    cell.classList.contains("pg-tile-pending")
      ? "pending"
      : (cell.querySelector("img")?.getAttribute("src") ?? "?"),
  );

// ── The gallery ────────────────────────────────────────────────────────────

it("gives a running generation a tile of its own, then the picture takes it", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Gallery");

  // In grid order: the lineage that has a picture, then the lineage that owes
  // one. A placeholder appended to the end of the wall would pass a count.
  await waitFor(() => expect(cells(container)).toEqual(["/playground/images/f1.png/thumb", "pending"]));

  const box = container.querySelector<HTMLElement>(".pg-tile-pending .pg-preview")!;
  // The run's own shape, reserved before anything has been decoded: this is what
  // the finished tile will be, so the columns are balanced once.
  expect(box.style.aspectRatio).toBe("1024 / 576");
  const reserved = box.style.aspectRatio;

  // The live preview, not a spinner: the same one-slot route the feed's cell
  // fetches, keyed by the sequence number off the SSE frame.
  await progress(generating(3));
  await waitFor(() =>
    expect(
      container.querySelector<HTMLImageElement>(".pg-tile-pending img")?.getAttribute("src"),
    ).toBe("/playground/api/preview?v=3"),
  );
  // And it says where the run is, with the cancel beside it.
  const tile = container.querySelector<HTMLElement>(".pg-tile-pending")!;
  expect(within(tile).getByText(/step 3 of 6/)).toBeTruthy();
  expect(within(tile).getByRole("button", { name: "Cancel" })).toBeTruthy();

  // The image lands.
  generations = [DONE, inFlight({ status: "completed", finishedAt: 3, images: [{ url: "/playground/images/f2.png", seed: 42 }] })];
  await progress({ state: "idle" });

  await waitFor(
    () =>
      expect(cells(container)).toEqual([
        "/playground/images/f1.png/thumb",
        "/playground/images/f2.png/thumb",
      ]),
    { timeout: 4000 },
  );
  // The Step 7 flicker fix, extended to the placeholder: the box the picture
  // arrived into is the box that was being held for it, so the wall does not
  // re-balance when one replaces the other.
  const landed = [...container.querySelectorAll<HTMLImageElement>(".pg-gallery .pg-thumb img")];
  expect(landed[1]!.style.aspectRatio).toBe(reserved);
  expect(landed[1]!.style.aspectRatio).toBe("1024 / 576");
});

it("holds one box per owed image and says the status once", async () => {
  // Found in a real browser, not here: an `n=2` request in flight drew the bar,
  // the step counter and a Cancel button on *both* of its placeholders, which
  // reads as two runs. A record has n boxes and one status.
  generations = [DONE, inFlight({ n: 2, seeds: [42, 43] })];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Gallery");

  await waitFor(() =>
    expect(cells(container)).toEqual(["/playground/images/f1.png/thumb", "pending", "pending"]),
  );
  await progress(generating(3));
  // Two boxes, one of them holding the live frame — the engine makes the images
  // of a request one after another, so only the first is being denoised.
  await waitFor(() =>
    expect(container.querySelectorAll(".pg-tile-pending .pg-preview img")).toHaveLength(1),
  );
  expect(container.querySelectorAll(".pg-tile-pending .pg-pending-note")).toHaveLength(1);
  expect(screen.getAllByRole("button", { name: "Cancel" })).toHaveLength(1);

  // And the filmstrip gives the record one frame rather than two identical empty
  // boxes, so the stage's default lands on the frame that has something to say.
  await user.click(screen.getByRole("tab", { name: "Light Table" }));
  await waitFor(() => expect(container.querySelector(".pg-table-pending")).toBeTruthy());
  expect(container.querySelectorAll(".pg-strip-pending")).toHaveLength(1);
  expect(within(container.querySelector<HTMLElement>(".pg-table-pending")!).getByText(/Image 1 of 2/)).toBeTruthy();
  await act(async () => {});
});

it("says the queue is holding a queued generation, in the gallery", async () => {
  paused = true;
  generations = [DONE, inFlight({ status: "queued", startedAt: null, finishedAt: null })];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Gallery");

  const tile = await waitFor(() => container.querySelector<HTMLElement>(".pg-tile-pending")!);
  // Held, not "Queued…": a queue that is not going to start this is a different
  // statement from one that will get to it.
  expect(within(tile).getByText("Held — the queue is paused.")).toBeTruthy();
  expect(within(tile).queryByText("Queued…")).toBeNull();
  await act(async () => {});
});

it("keeps a failed and a cancelled run visible in the gallery", async () => {
  generations = [
    DONE,
    inFlight({ status: "failed", error: "Out of memory", finishedAt: 3 }),
    inFlight({
      id: "g3",
      groupId: "g3",
      status: "cancelled",
      prompt: "a fox in the rain",
      finishedAt: 3,
    }),
  ];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Gallery");

  // Three cells for three lineages: a run that ended badly and left no picture
  // must not silently leave the view either.
  await waitFor(() =>
    expect(cells(container)).toEqual(["/playground/images/f1.png/thumb", "pending", "pending"]),
  );
  expect(screen.getByText("Generation failed.")).toBeTruthy();
  expect(screen.getByText(/Out of memory/)).toBeTruthy();
  expect(screen.getByText("Cancelled.")).toBeTruthy();
  await act(async () => {});
});

it("cancels from the gallery through the request the feed sends", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  server.on("POST /playground/api/generations/g2/cancel", () => ({
    ...inFlight({ status: "cancelled", finishedAt: 3 }),
  }));
  await open(user, "Gallery");
  await waitFor(() => expect(container.querySelector(".pg-tile-pending")).toBeTruthy());

  await user.click(screen.getByRole("button", { name: "Cancel" }));

  // The request, not the click: a button wired to nothing, or to a second
  // client-side notion of cancelling, would pass an assertion about the click.
  // The generation's id is in the path, which is the identity that matters —
  // cancelling the run that is on screen and not merely "a run".
  await waitFor(() =>
    expect(
      server.requests.filter(
        (request) =>
          request.method === "POST" && request.path === "/playground/api/generations/g2/cancel",
      ),
    ).toHaveLength(1),
  );
  await act(async () => {});
});

// ── The light table ────────────────────────────────────────────────────────

it("gives the filmstrip a frame for a running generation, and the stage to it", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Light Table");

  // Two frames: the picture, and the run that owes one.
  await waitFor(() => expect(container.querySelectorAll(".pg-strip-tile")).toHaveLength(2));
  expect(container.querySelectorAll(".pg-strip-pending")).toHaveLength(1);

  // The stage is the run, not the old picture: a hero sitting on last week's fox
  // with no explanation while a request is running is the reported defect at its
  // worst, because it looks like an answer.
  expect(container.querySelector(".pg-table-pending")).toBeTruthy();
  expect(container.querySelector(".pg-table-hero")).toBeNull();
  await progress(generating(2));
  await waitFor(() =>
    expect(
      container.querySelector<HTMLImageElement>(".pg-table-pending img")?.getAttribute("src"),
    ).toBe("/playground/api/preview?v=2"),
  );
  // The panel states the run's own settings, and no seed: a seed belongs to an
  // image, and this request has not produced one.
  const panel = within(screen.getByRole("complementary", { name: "Image details" }));
  expect(panel.getByText("a fox at dusk")).toBeTruthy();
  expect(panel.getByText("1024x576")).toBeTruthy();
  expect(panel.queryByText("Seed")).toBeNull();
  // And the cancel is here too, so watching a run in this view is not a trip
  // back to the feed to stop it.
  expect(screen.getAllByRole("button", { name: "Cancel" }).length).toBeGreaterThan(0);

  // The image lands: the placeholder frame becomes a picture like any other.
  generations = [DONE, inFlight({ status: "completed", finishedAt: 3, images: [{ url: "/playground/images/f2.png", seed: 42 }] })];
  await progress({ state: "idle" });

  await waitFor(() => expect(container.querySelectorAll(".pg-strip-pending")).toHaveLength(0), {
    timeout: 4000,
  });
  expect(
    [...container.querySelectorAll<HTMLImageElement>(".pg-strip-tile img")].map((image) =>
      image.getAttribute("src"),
    ),
  ).toEqual(["/playground/images/f1.png/thumb", "/playground/images/f2.png/thumb"]);
  expect(container.querySelector(".pg-table-pending")).toBeNull();
});

it("says the queue is holding a queued generation, in the light table", async () => {
  paused = true;
  generations = [DONE, inFlight({ status: "queued", startedAt: null, finishedAt: null })];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Light Table");

  await waitFor(() => expect(container.querySelector(".pg-strip-pending")).toBeTruthy());
  expect(screen.getByText("Held — the queue is paused.")).toBeTruthy();
  expect(screen.queryByText("Queued…")).toBeNull();
  await act(async () => {});
});

it("keeps a failed run visible in the light table", async () => {
  generations = [DONE, inFlight({ status: "failed", error: "Out of memory", finishedAt: 3 })];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Light Table");

  // Still visible, and that is what this test was written for: a run that
  // produced nothing must not vanish from the project.
  await waitFor(() => expect(container.querySelector(".pg-strip-pending")).toBeTruthy());

  // Re-aimed, on the user's call, after they hit it: this used to require the
  // failure on the *stage* too — "the stage says so rather than showing the
  // previous picture as though nothing had happened". Reported from a real
  // project where that rule put "Generation failed. Interrupted by server
  // restart" on the stage while twelve finished pictures sat in the strip.
  //
  // The failure keeps its frame; the stage keeps a picture. Nothing is hidden —
  // the marker is one click away and the strip is where the project's history
  // lives — and the view opens on something worth looking at.
  expect(container.querySelector(".pg-table-pending")).toBeNull();
  expect(container.querySelector(".pg-table-hero img")).toBeTruthy();

  // The words themselves are still rendered — on the strip's marker, which is
  // where a failure now lives. Announced with its reason rather than as
  // "Generating:", which is what that tile used to claim whatever had happened.
  // `listitem`, not `button`: the tile carries an explicit `role` so the strip
  // reads as a list, and the implicit button role is overridden.
  expect(
    screen.getByRole("listitem", { name: /Generation failed: Out of memory/ }),
  ).toBeTruthy();
  await act(async () => {});
});

it("leaves a project with nothing in flight exactly as it was", async () => {
  // The other half of every assertion above: none of this may appear for a
  // project whose runs have all finished. A view that drew a placeholder for a
  // completed record would be showing work that is not in flight.
  generations = [DONE];
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await open(user, "Gallery");
  await waitFor(() => expect(cells(container)).toEqual(["/playground/images/f1.png/thumb"]));

  await user.click(screen.getByRole("tab", { name: "Light Table" }));
  await waitFor(() => expect(container.querySelector(".pg-table-hero")).toBeTruthy());
  expect(container.querySelector(".pg-strip-pending")).toBeNull();
  expect(container.querySelector(".pg-table-pending")).toBeNull();
  await act(async () => {});
});
