/**
 * The gallery, and the switch between it and the prompt feed.
 *
 * Three things are witnessed here, and the first two are about *identity* rather
 * than about counts:
 *
 * - the gallery shows exactly the images the feed shows for the same project,
 *   flattened — so an implementation that drew only the first lineage's images,
 *   or that leaked another project's, fails even though it renders the right
 *   *number* of tiles for the fixtures it happens to be given;
 * - the tiles load the thumbnail route and nothing loads the full file until a
 *   tile is opened;
 * - switching is presentation only: it issues no mutating request, and the
 *   remembered choice is a preference that can be lost without taking the
 *   project with it.
 */

import { render, screen, waitFor } from "@testing-library/react";
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
  model: "qwen-image-2512",
  steps: 6,
  contextImage: null,
  status: "completed" as const,
  error: null,
  createdAt: 1,
  startedAt: 1,
  finishedAt: 2,
};

/** Two lineages in one project, and an upscale that joins the first. */
const FOXES: PlaygroundGeneration[] = [
  {
    ...base,
    id: "g1",
    sessionId: "s1",
    groupId: "g1",
    prompt: "a fox in the snow",
    kind: "txt2img",
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
    // The upscale belongs to the first lineage, so its image is part of that
    // group — the case a naive "images of the first generation" flattening gets
    // wrong in the other direction.
    groupId: "g1",
    prompt: "a fox in the snow",
    kind: "upscale",
    n: 1,
    size: "1024x1024",
    seeds: [2],
    images: [{ url: "/playground/images/f2-upscaled.png", seed: 2 }],
  },
  {
    ...base,
    id: "g3",
    sessionId: "s1",
    groupId: "g3",
    prompt: "a fox at dusk",
    kind: "txt2img",
    n: 1,
    // Landscape, unlike its siblings: the tiles reserve their boxes from this
    // string, so a fixture of all-square runs could not tell a real derivation
    // from a constant.
    size: "1024x576",
    seeds: [3],
    images: [{ url: "/playground/images/f3.png", seed: 3 }],
  },
];

/** Another project entirely. None of this may ever appear in s1's gallery. */
const BADGERS: PlaygroundGeneration[] = [
  {
    ...base,
    id: "b1",
    sessionId: "s2",
    groupId: "b1",
    prompt: "a badger",
    kind: "txt2img",
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
  "/playground/images/f3.png",
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
 * preference has to bring its own store. `PlaygroundApp.projects.test.tsx` does
 * the same for the rail.
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

/** Every picture on screen, in document order, by the URL it is loading. */
const shown = (container: HTMLElement, selector: string) =>
  [...container.querySelectorAll<HTMLImageElement>(selector)].map((image) =>
    image.getAttribute("src"),
  );

async function openGallery(user: UserEvent) {
  await screen.findAllByAltText("a fox in the snow");
  await user.click(screen.getByRole("tab", { name: "Gallery" }));
}

it("shows exactly the project's own images, flattened out of their lineages", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);

  // What the feed shows, taken from the feed itself rather than restated: the
  // property under test is that the two views agree, so the expectation has to
  // come from one of them.
  await screen.findAllByAltText("a fox in the snow");
  const inFeed = shown(container, ".pg-images .pg-thumb img");
  expect(inFeed).toEqual(FOX_IMAGES);

  await user.click(screen.getByRole("tab", { name: "Gallery" }));
  await waitFor(() => expect(container.querySelector(".pg-gallery")).toBeTruthy());

  // The same images, the same order, as tiles. Identity, not count: a gallery
  // that drew the first lineage only, or dropped the upscale, or reordered,
  // fails here.
  expect(shown(container, ".pg-gallery .pg-thumb img")).toEqual(
    FOX_IMAGES.map((url) => `${url}/thumb`),
  );
  // And nothing of the other project, in either form.
  expect(container.innerHTML).not.toContain("b1.png");
});

it("loads thumbnails, lazily, and fetches a full file only when a tile is opened", async () => {
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openGallery(user);

  const tiles = [...container.querySelectorAll<HTMLImageElement>(".pg-gallery .pg-thumb img")];
  expect(tiles).toHaveLength(FOX_IMAGES.length);
  for (const tile of tiles) {
    // The derived tile, never the file: ~22 KB against ~1.9 MB, measured.
    expect(tile.getAttribute("src")?.endsWith("/thumb")).toBe(true);
    expect(FOX_IMAGES).not.toContain(tile.getAttribute("src"));
    expect(tile.getAttribute("loading")).toBe("lazy");
  }

  // Opening one is the one thing that asks for the full-resolution file.
  await user.click(screen.getAllByAltText("a fox in the snow")[0]!);
  const viewer = await screen.findByRole("dialog");
  expect(viewer.querySelector<HTMLImageElement>(".pg-viewer-image")?.getAttribute("src")).toBe(
    "/playground/images/f1.png",
  );
});

it("reserves each tile's box from its run's size, before any picture loads", async () => {
  // The flicker the user reported: tiles were `width: 100%; height: auto`, so
  // the column balancer had nothing to work with until every intrinsic height
  // had been decoded — the wall re-flowed as images arrived and again on every
  // frame of a window resize. Nothing has loaded in jsdom at all, which is
  // precisely the state under test: the shape has to be known anyway.
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openGallery(user);

  const ratios = [...container.querySelectorAll<HTMLImageElement>(".pg-gallery .pg-thumb img")].map(
    (image) => image.style.aspectRatio,
  );
  // Each image's *own* run, not the project's first: the upscale is 1024², and
  // the last lineage is landscape. A constant, or the root's size read for
  // every tile in its lineage, fails here.
  expect(ratios).toEqual(["512 / 512", "512 / 512", "1024 / 1024", "1024 / 576"]);
});

it("carries the session's unlock token on the thumbnail URLs", async () => {
  // A locked project's tiles are refused to exactly the caller its full images
  // are refused to, and an `<img>` sends no headers — so the token has to be on
  // the URL here as well, or a locked project's gallery is a wall of 403s.
  window.sessionStorage.setItem("qds.playground.unlock.s1", "tok-123");
  server.on("GET /playground/api/sessions", () => ({
    paused: false,
    sessions: [playgroundSession({ id: "s1", title: "foxes", locked: true })],
  }));
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await openGallery(user);

  expect(shown(container, ".pg-gallery .pg-thumb img")).toEqual(
    FOX_IMAGES.map((url) => `${url}/thumb?t=tok-123`),
  );
});

it("switching view mutates nothing and asks the server for nothing", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findAllByAltText("a fox in the snow");
  const before = server.requests.length;

  await user.click(screen.getByRole("tab", { name: "Gallery" }));
  await waitFor(() => expect(screen.queryByText("a fox in the snow")).toBeNull());
  await user.click(screen.getByRole("tab", { name: "Prompts" }));
  await screen.findByText("a fox in the snow");

  // Not "no POST": no request at all. The images are already in hand, and a
  // presentation change that refetched the project would also be a presentation
  // change that could fail.
  expect(server.requests.slice(before)).toEqual([]);
  expect(server.requests.every((request) => request.method === "GET")).toBe(true);
});

it("puts the view in the URL as ?mode=, leaving ?view= to the surface", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findAllByAltText("a fox in the snow");

  await user.click(screen.getByRole("tab", { name: "Gallery" }));
  await waitFor(() =>
    expect(new URLSearchParams(window.location.search).get("mode")).toBe("gallery"),
  );
  // The project is still in the URL beside it, and nothing claimed `view`.
  expect(new URLSearchParams(window.location.search).get("session")).toBe("s1");
  expect(new URLSearchParams(window.location.search).get("view")).toBeNull();
});

it("opens a project in the view it was left in, per project", async () => {
  const store = new Map<string, string>();
  const restore = withStorage({
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => void store.set(key, value),
  });
  try {
    const user = userEvent.setup();
    const { container } = render(<PlaygroundApp />);
    await openGallery(user);
    expect(store.get("qds.playground.view.s1")).toBe("gallery");

    // The other project has no preference of its own: it opens in the default.
    await user.click(screen.getByRole("button", { name: /^badgers/ }));
    await screen.findByText("a badger");
    expect(container.querySelector(".pg-gallery")).toBeNull();

    // Back to the first, which is where it was left.
    await user.click(screen.getByRole("button", { name: /^foxes/ }));
    await waitFor(() => expect(container.querySelector(".pg-gallery")).toBeTruthy());
    expect(screen.queryByText("a fox in the snow")).toBeNull();

    // And the second is still in its own state, not in the first's.
    await user.click(screen.getByRole("button", { name: /^badgers/ }));
    await screen.findByText("a badger");
    expect(container.querySelector(".pg-gallery")).toBeNull();
  } finally {
    restore();
  }
});

it("opens the project in the default view when the preference cannot be read", async () => {
  // Safari's private mode throws on write; a quota refusal throws too. Either
  // way the project opens — the preference is the only thing that is lost.
  const restore = withStorage({
    getItem: () => {
      throw new Error("storage disabled");
    },
    setItem: () => {
      throw new Error("storage disabled");
    },
  });
  try {
    const user = userEvent.setup();
    const { container } = render(<PlaygroundApp />);
    await screen.findByText("a fox in the snow");

    // And switching still works, it just is not remembered.
    await user.click(screen.getByRole("tab", { name: "Gallery" }));
    await waitFor(() => expect(container.querySelector(".pg-gallery")).toBeTruthy());
  } finally {
    restore();
  }
});

it("opens the project in the default view when the preference is junk", async () => {
  const restore = withStorage({
    getItem: () => "light-table-someday",
    setItem: () => {},
  });
  try {
    render(<PlaygroundApp />);
    // An unknown value reads as "no preference", not as "some third view".
    await screen.findByText("a fox in the snow");
    expect(screen.getByRole("tab", { name: "Prompts" }).getAttribute("aria-selected")).toBe(
      "true",
    );
  } finally {
    restore();
  }
});

it("honours ?mode= from a followed link, once, for the project it arrived with", async () => {
  window.history.replaceState(null, "", "/playground/?session=s1&mode=gallery");
  const user = userEvent.setup();
  const { container } = render(<PlaygroundApp />);
  await waitFor(() => expect(container.querySelector(".pg-gallery")).toBeTruthy());

  // Spent: the next project selected gets its own preference, or the default —
  // not the one the link asked for.
  await user.click(screen.getByRole("button", { name: /^badgers/ }));
  await screen.findByText("a badger");
  expect(container.querySelector(".pg-gallery")).toBeNull();
});
