/**
 * The light table: one picture on the stage, the project as a strip, the facts
 * of the *selected* image beside it.
 *
 * The first test is the one this view exists to get right, and it is written to
 * fail against the obvious implementation rather than against a typo. An upscale
 * joins its source's lineage, so the 4096² picture below sits in a group whose
 * root asked a diffusion model for 512². An inspector wired to `group.root` —
 * which is where the prompt legitimately comes from, and therefore the tempting
 * place to read everything from — labels that picture 512x512 and
 * `qwen-image-2512`. Both assertions here discriminate that: the panel must show
 * the upscaler and the enlarged size, and must not show the root's.
 *
 * The rest witness the properties the view shares with the gallery (thumbnails
 * in the strip, the full file on the stage, the unlock token on both, a switch
 * that costs no request, a preference remembered per project) plus the one thing
 * it adds: selection, which is view-local and therefore must not follow the user
 * into another project.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { type FakeServer, fakeServer } from "../test-server";
import type { PlaygroundGeneration } from "../types";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;

const base = {
  negativePrompt: null,
  rewrittenPrompt: null,
  rewriteError: null,
  contextImage: null,
  status: "completed" as const,
  error: null,
  createdAt: 1,
  startedAt: 1,
  finishedAt: 2,
};

/**
 * One lineage: a small txt2img, and the ×8 upscale of its second image.
 *
 * Every field the inspector shows differs between the two records — size, model
 * and step count — which is what makes "the selected image's, not the root's"
 * an assertion rather than a coincidence. The upscale's `steps: 0` is the
 * server's own value (`admission.submit_upscale`): a super-resolution pass has
 * no denoising loop.
 */
const FOXES: PlaygroundGeneration[] = [
  {
    ...base,
    id: "g1",
    sessionId: "s1",
    groupId: "g1",
    prompt: "a fox in the snow",
    rewrittenPrompt: "a red fox curled in deep powder, low winter sun",
    kind: "txt2img",
    model: "qwen-image-2512",
    steps: 6,
    n: 2,
    size: "512x512",
    seeds: [1, 2],
    images: [
      { url: "/playground/images/f1.png", seed: 1 },
      { url: "/playground/images/f2.png", seed: 2 },
    ],
  },
  {
    ...base,
    id: "g2",
    sessionId: "s1",
    // The upscale joins the lineage it came from, which is the whole trap.
    groupId: "g1",
    prompt: "a fox in the snow",
    kind: "upscale",
    model: "realesrgan-x4plus",
    steps: 0,
    n: 1,
    size: "4096x4096",
    seeds: [2],
    images: [{ url: "/playground/images/f2-upscaled.png", seed: 2 }],
  },
];

/** Another project entirely, to switch away to. */
const BADGERS: PlaygroundGeneration[] = [
  {
    ...base,
    id: "b1",
    sessionId: "s2",
    groupId: "b1",
    prompt: "a badger",
    kind: "txt2img",
    model: "qwen-image-2512",
    steps: 6,
    n: 1,
    size: "512x512",
    seeds: [9],
    images: [{ url: "/playground/images/b1.png", seed: 9 }],
  },
];

const FOX_IMAGES = [
  "/playground/images/f1.png",
  "/playground/images/f2.png",
  "/playground/images/f2-upscaled.png",
];

beforeEach(() => {
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({ max_n: 4, default_model: "qwen-image-2512" }));
  server.on("GET /v1/progress", () => new Response(new ReadableStream({ start() {} })));
  server.on("GET /playground/api/upscalers", () => ({ upscalers: [] }));
  server.on("GET /playground/api/sessions", () => ({
    paused: false,
    sessions: [
      playgroundSession({ id: "s1", title: "foxes" }),
      playgroundSession({ id: "s2", title: "badgers" }),
    ],
  }));
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: FOXES,
  }));
  server.on("GET /playground/api/sessions/s2", () => ({
    session: playgroundSession({ id: "s2", title: "badgers" }),
    generations: BADGERS,
  }));
  window.history.replaceState(null, "", "/playground/?session=s1");
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
  vi.restoreAllMocks();
});

/**
 * A `Storage`-shaped seam, installed for one test.
 *
 * jsdom's `localStorage` is shadowed by Node's own under Node >= 22 and reads
 * back `undefined` (see `test-setup.ts`), so a test about a remembered
 * preference has to bring its own store — as the gallery's and the projects'
 * tests do.
 */
function withStorage(store: {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
}): () => void {
  const stub = { ...store, removeItem: () => {}, clear: () => {}, key: () => null, length: 0 };
  const original = Object.getOwnPropertyDescriptor(window, "localStorage");
  Object.defineProperty(window, "localStorage", { value: stub, configurable: true });
  return () => {
    if (original) Object.defineProperty(window, "localStorage", original);
    else Reflect.deleteProperty(window, "localStorage");
  };
}

/** What the stage is showing, by the file it is loading. */
const onStage = (container: HTMLElement) =>
  container.querySelector<HTMLImageElement>(".pg-table-hero img")?.getAttribute("src");

/** The strip, in document order, by the URL each tile is loading. */
const inStrip = (container: HTMLElement) =>
  [...container.querySelectorAll<HTMLImageElement>(".pg-strip-tile img")].map((image) =>
    image.getAttribute("src"),
  );

const inspector = () => screen.getByRole("complementary", { name: "Image details" });

async function openTable(user: UserEvent) {
  await screen.findAllByAltText("a fox in the snow");
  await user.click(screen.getByRole("tab", { name: "Light Table" }));
  await waitFor(() => expect(document.querySelector(".pg-table")).toBeTruthy());
}

it("shows the selected image's own settings, not its group root's", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  // The first frame is the root's own first image, so the panel legitimately
  // reads the root's numbers here — establishing that the values differ, so the
  // assertions below are about *whose* they are, not about which exist.
  expect(within(inspector()).getByText("512x512")).toBeTruthy();
  expect(within(inspector()).getByText("qwen-image-2512")).toBeTruthy();
  expect(within(inspector()).getByText("6")).toBeTruthy();

  // Now the upscale — third tile, last in the lineage.
  await user.click(container.querySelectorAll<HTMLElement>(".pg-strip-tile")[2]!);

  const panel = within(inspector());
  // Its own facts.
  expect(panel.getByText("4096x4096")).toBeTruthy();
  expect(panel.getByText("realesrgan-x4plus")).toBeTruthy();
  // And not its ancestor's, which is the failure `GroupImage` exists to prevent:
  // reading `group.root` here labels a 4096² picture 512² with a diffusion model
  // that never touched it.
  expect(panel.queryByText("512x512")).toBeNull();
  expect(panel.queryByText("qwen-image-2512")).toBeNull();
  // No step count either: the server records `steps=0` for an upscale, and the
  // root's 6 belongs to a sampler that did not produce this file.
  expect(panel.queryByText("Steps")).toBeNull();
  expect(panel.queryByText("6")).toBeNull();
  // The seed is the source's, which is the honest value the server stored.
  expect(panel.getByText("2")).toBeTruthy();

  // The prompt still comes from the root, deliberately: an upscale has no prompt
  // of its own, and the lineage's prompt is the lineage's.
  expect(panel.getByText("a fox in the snow")).toBeTruthy();
  expect(panel.getByText("a red fox curled in deep powder, low winter sun")).toBeTruthy();
});

it("puts the picked tile on the stage", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  // Opens on the project's first image, and the strip says which one that is.
  expect(onStage(container)).toBe(FOX_IMAGES[0]);
  expect(
    container.querySelectorAll(".pg-strip-tile")[0]!.getAttribute("aria-current"),
  ).toBe("true");

  await user.click(container.querySelectorAll<HTMLElement>(".pg-strip-tile")[1]!);

  expect(onStage(container)).toBe(FOX_IMAGES[1]);
  const tiles = container.querySelectorAll(".pg-strip-tile");
  expect(tiles[1]!.getAttribute("aria-current")).toBe("true");
  expect(tiles[0]!.getAttribute("aria-current")).toBeNull();

  // And the arrows move along the same strip.
  await user.click(screen.getByRole("button", { name: "Next image" }));
  expect(onStage(container)).toBe(FOX_IMAGES[2]);
  await user.click(screen.getByRole("button", { name: "Previous image" }));
  expect(onStage(container)).toBe(FOX_IMAGES[1]);
});

it("loads the strip from thumbnails, lazily, and the stage from the file", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  // The whole project is in the strip, in the feed's order, as derived tiles:
  // ~22 KB each against ~1.9 MB, measured. A filmstrip of full-resolution files
  // is what makes this view unaffordable on a hundred-image project.
  expect(inStrip(container)).toEqual(FOX_IMAGES.map((url) => `${url}/thumb`));
  for (const tile of container.querySelectorAll<HTMLImageElement>(".pg-strip-tile img")) {
    expect(tile.getAttribute("loading")).toBe("lazy");
  }
  // The one picture being looked at is the one full file fetched.
  expect(onStage(container)).toBe(FOX_IMAGES[0]);
  expect(onStage(container)?.endsWith("/thumb")).toBe(false);
});

it("carries the session's unlock token on the stage and on every tile", async () => {
  // An `<img>` sends no headers, so a locked project's pictures are reachable
  // only if the token is on the URL — on both routes, because both are refused
  // to exactly the caller the other is refused to.
  window.sessionStorage.setItem("qds.playground.unlock.s1", "tok-123");
  server.on("GET /playground/api/sessions", () => ({
    paused: false,
    sessions: [playgroundSession({ id: "s1", title: "foxes", locked: true })],
  }));
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  expect(onStage(container)).toBe(`${FOX_IMAGES[0]}?t=tok-123`);
  expect(inStrip(container)).toEqual(FOX_IMAGES.map((url) => `${url}/thumb?t=tok-123`));
});

it("switching to the light table and back mutates nothing and asks for nothing", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findAllByAltText("a fox in the snow");
  const before = server.requests.length;

  await user.click(screen.getByRole("tab", { name: "Light Table" }));
  await waitFor(() => expect(document.querySelector(".pg-table")).toBeTruthy());
  await user.click(screen.getByRole("tab", { name: "Prompts" }));
  await screen.findByText("a fox in the snow");

  // Not "no POST": no request at all. T1 — a view switch is presentation, and
  // the images are already in hand.
  expect(server.requests.slice(before)).toEqual([]);
  expect(server.requests.every((request) => request.method === "GET")).toBe(true);
});

it("remembers the light table per project, and puts it in the URL", async () => {
  const store = new Map<string, string>();
  const restore = withStorage({
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => void store.set(key, value),
  });
  try {
    const user = userEvent.setup();
    const { container } = render(<PlaygroundApp />);
    await openTable(user);
    expect(store.get("qds.playground.view.s1")).toBe("table");
    expect(new URLSearchParams(window.location.search).get("mode")).toBe("table");

    // The other project has no preference: it opens in the default.
    await user.click(screen.getByRole("button", { name: /^badgers/ }));
    await screen.findByText("a badger");
    expect(container.querySelector(".pg-table")).toBeNull();

    // Back to the first, which is where it was left.
    await user.click(screen.getByRole("button", { name: /^foxes/ }));
    await waitFor(() => expect(container.querySelector(".pg-table")).toBeTruthy());
  } finally {
    restore();
  }
});

it("does not carry the selected image into another project", async () => {
  // Selection is this view's own state, not a fact about the project: it is not
  // in `PlaygroundApp`'s server-owned state and nothing persists it. Switching
  // projects must therefore land on the new project's first image — never on an
  // index into a strip that has changed underneath it.
  const store = new Map<string, string>([["qds.playground.view.s2", "table"]]);
  const restore = withStorage({
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => void store.set(key, value),
  });
  try {
    const user = userEvent.setup();
    const { container } = render(<PlaygroundApp />);
    await openTable(user);
    await user.click(container.querySelectorAll<HTMLElement>(".pg-strip-tile")[2]!);
    expect(onStage(container)).toBe(FOX_IMAGES[2]);

    await user.click(screen.getByRole("button", { name: /^badgers/ }));
    await waitFor(() => expect(inStrip(container)).toHaveLength(1));
    expect(onStage(container)).toBe("/playground/images/b1.png");

    // And back: the first image again, not the third of a strip that no longer
    // has three.
    await user.click(screen.getByRole("button", { name: /^foxes/ }));
    await waitFor(() => expect(inStrip(container)).toHaveLength(3));
    expect(onStage(container)).toBe(FOX_IMAGES[0]);
  } finally {
    restore();
  }
});

it("opens the stage's picture in the viewer, full resolution", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);
  await user.click(container.querySelectorAll<HTMLElement>(".pg-strip-tile")[2]!);

  await user.click(container.querySelector<HTMLElement>(".pg-table-hero")!);
  const viewer = await screen.findByRole("dialog");

  expect(viewer.querySelector<HTMLImageElement>(".pg-viewer-image")?.getAttribute("src")).toBe(
    FOX_IMAGES[2],
  );
  // The footer states the *image's* facts, through the same `detailOf` both
  // other views use — so the upscale reads as upscaled, at its own size.
  expect(viewer.textContent).toContain("4096x4096");
});

it("opens on the first picture, not on a failed run that came before it", async () => {
  // Reported from the browser twice before it was read as a defect: a project
  // whose first lineage failed opened with "Generation failed" on the stage
  // while a dozen finished pictures sat in the strip. A failure is a frame —
  // it has to be, or a run that produced nothing would vanish — but it is not
  // something to look at, and the stage is for looking.
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [
      {
        ...base,
        id: "g0",
        sessionId: "s1",
        groupId: "g0",
        prompt: "a fox that never arrived",
        kind: "txt2img" as const,
        model: "qwen-image-2512",
        steps: 6,
        n: 1,
        size: "512x512",
        seeds: [9],
        images: [],
        status: "failed" as const,
        error: "Interrupted by server restart",
      },
      ...FOXES,
    ],
  }));

  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  // The failure keeps its place in the strip — first, where it happened.
  // Counted as tiles, not as `<img>`: a failed run has no picture to load, so
  // `inStrip` (which reads image sources) cannot see it.
  await waitFor(() =>
    expect(container.querySelectorAll(".pg-strip-tile")).toHaveLength(4),
  );
  // …and the stage holds the first actual picture instead.
  expect(onStage(container)).toBe(FOX_IMAGES[0]);
  expect(container.querySelector(".pg-table-hero img")).toBeTruthy();
});

it("still shows the failure when a project has nothing else", async () => {
  // The counterfactual, and the reason the fix is "first picture" rather than
  // "skip failures": with no picture to prefer, the view must say what happened
  // rather than render an empty stage.
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [
      {
        ...base,
        id: "g0",
        sessionId: "s1",
        groupId: "g0",
        prompt: "a fox that never arrived",
        kind: "txt2img" as const,
        model: "qwen-image-2512",
        steps: 6,
        n: 1,
        size: "512x512",
        seeds: [9],
        images: [],
        status: "failed" as const,
        error: "Interrupted by server restart",
      },
    ],
  }));

  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await user.click(screen.getByRole("tab", { name: "Light Table" }));
  await waitFor(() => expect(document.querySelector(".pg-table")).toBeTruthy());

  expect(container.textContent).toContain("Interrupted by server restart");
});

it("does not treat a failed run as a run in flight when it is the newest", async () => {
  // The case the first fix missed, and the one the user actually had. A frame
  // is `pending` whenever its record owes no file, so `failed` and `cancelled`
  // look exactly like `queued` and `running` unless the status is read. With
  // the failure LAST, "prefer the newest when something is being made" grabbed
  // it and pinned the stage to "Generation failed" — measured in the browser as
  // aria-current on the sixteenth of sixteen tiles.
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [
      ...FOXES,
      {
        ...base,
        id: "g9",
        sessionId: "s1",
        groupId: "g9",
        prompt: "a fox that never arrived",
        kind: "txt2img" as const,
        model: "qwen-image-2512",
        steps: 6,
        n: 1,
        size: "512x512",
        seeds: [9],
        images: [],
        status: "failed" as const,
        error: "Interrupted by server restart",
      },
    ],
  }));

  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  await waitFor(() =>
    expect(container.querySelectorAll(".pg-strip-tile")).toHaveLength(4),
  );
  // A picture on the stage, not the failure that happened after it.
  expect(container.querySelector(".pg-table-pending")).toBeNull();
  expect(onStage(container)).toBe(FOX_IMAGES[0]);
});

it("still gives the stage to a run that is genuinely running", async () => {
  // The counterfactual for the fix above: narrowing "in flight" to queued and
  // running must not cost the property it was written for.
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [
      ...FOXES,
      {
        ...base,
        id: "g9",
        sessionId: "s1",
        groupId: "g9",
        prompt: "a fox still arriving",
        kind: "txt2img" as const,
        model: "qwen-image-2512",
        steps: 6,
        n: 1,
        size: "512x512",
        seeds: [9],
        images: [],
        status: "running" as const,
        error: null,
        finishedAt: null,
      },
    ],
  }));

  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openTable(user);

  await waitFor(() => expect(container.querySelector(".pg-table-pending")).toBeTruthy());
  expect(onStage(container)).toBeUndefined();
});
