/**
 * `?view=plugin` — the studio without the controls around it.
 *
 * Hermes' `qds-playground` plugin creates the session and drives generation
 * from the chat, then opens this URL in its preview pane. What it must show is
 * the work: the feed, and the live preview inside it. What it must NOT show is
 * a second way to do what the chat already does — the project rail and the
 * prompt composer — because two surfaces owning the same action is how they
 * drift apart.
 *
 * The default view is the contract's other half: without `?view=plugin`
 * nothing here changes, so the plain playground keeps every control.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { type FakeServer, fakeServer } from "../test-server";
import type { PlaygroundGeneration } from "../types";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;

const GENERATION: PlaygroundGeneration = {
  id: "g1",
  sessionId: "s1",
  groupId: "g1",
  prompt: "a lighthouse",
  negativePrompt: null,
  rewrittenPrompt: null,
  rewriteError: null,
  model: "qwen-image-2512",
  kind: "txt2img",
  n: 1,
  size: "512x288",
  steps: 6,
  seeds: [41],
  contextImage: null,
  status: "completed",
  error: null,
  images: [{ url: "/playground/images/g1.png", seed: 41 }],
  createdAt: 1,
  startedAt: 1,
  finishedAt: 2,
};

beforeEach(() => {
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({
    max_n: 4,
    default_model: "qwen-image-2512",
  }));
  server.on(
    "GET /v1/progress",
    () => new Response(new ReadableStream({ start() {} })),
  );
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "lighthouses" })],
  }));
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "lighthouses" }),
    generations: [GENERATION],
  }));
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
  vi.restoreAllMocks();
});

/** The feed itself: proof the studio rendered, not just the page shell. */
async function feedIsShowing() {
  await waitFor(() => expect(screen.getByAltText("a lighthouse")).toBeTruthy());
}

it("hides the session sidebar and the composer under ?view=plugin", async () => {
  window.history.replaceState(null, "", "/playground/?session=s1&view=plugin");
  render(<PlaygroundApp />);

  await feedIsShowing();

  // The two surfaces the chat replaces.
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(screen.queryByRole("button", { name: /generate/i })).toBeNull();
  // No rail: not the list, not the "New project" control, not the collapse
  // toggle. Asked of the landmark element as well as of the labels, because a
  // label is the part most likely to be renamed — as "New session" just was —
  // and a query for a name nobody renders passes by being wrong twice.
  expect(screen.queryByRole("complementary")).toBeNull();
  expect(screen.queryByRole("button", { name: /new project/i })).toBeNull();
  expect(screen.queryByRole("button", { name: /project rail/i })).toBeNull();
  // No view switcher either, and no `?mode=` written into the embedder's URL:
  // this surface renders one view, and which one is the pane owner's choice —
  // not a control the chat has to explain.
  expect(screen.queryByRole("tablist")).toBeNull();
  expect(screen.queryByRole("tab", { name: /gallery/i })).toBeNull();
  expect(screen.queryByRole("tab", { name: /light table/i })).toBeNull();
  expect(new URLSearchParams(window.location.search).get("mode")).toBeNull();
  // What it must NOT lose with the rail: the queue's state at a glance. The
  // pill moved into the rail's footer with the redesign, and this surface has
  // no rail — so it is rendered in the masthead here instead. A pane with no
  // sidebar to expand and no browser chrome cannot go looking for it.
  expect(screen.getByText("Running")).toBeTruthy();
});

it("shows a held queue in the masthead when embedded, where the rail is not", async () => {
  // The list payload is what carries `paused` — the same one the rail's footer
  // reads on the full surface.
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "lighthouses" })],
    paused: true,
  }));
  window.history.replaceState(null, "", "/playground/?session=s1&view=plugin");
  render(<PlaygroundApp />);

  expect(await screen.findByText("Queue paused")).toBeTruthy();
  // Still no rail — the pill is in the masthead, not a sidebar smuggled back in.
  expect(screen.queryByRole("complementary")).toBeNull();
  // Nowhere to navigate from a pane with no browser chrome.
  expect(screen.queryByRole("link", { name: /server config/i })).toBeNull();
});

it("keeps the queue control under ?view=plugin", async () => {
  // Pausing is a server-wide state the embedder does not own, and a held queue
  // explains a generation that is not starting — so this one stays.
  window.history.replaceState(null, "", "/playground/?session=s1&view=plugin");
  render(<PlaygroundApp />);

  await feedIsShowing();

  expect(screen.getByRole("button", { name: /pause queue/i })).toBeTruthy();
});

it("keeps every control in the default view", async () => {
  window.history.replaceState(null, "", "/playground/?session=s1");
  render(<PlaygroundApp />);

  await feedIsShowing();

  expect(screen.getByRole("textbox")).toBeTruthy();
  expect(screen.getByRole("link", { name: /server config/i })).toBeTruthy();
  expect(screen.getByRole("tab", { name: "Prompts" })).toBeTruthy();
  expect(screen.getByRole("tab", { name: "Gallery" })).toBeTruthy();
  expect(screen.getByRole("tab", { name: "Light Table" })).toBeTruthy();
});
