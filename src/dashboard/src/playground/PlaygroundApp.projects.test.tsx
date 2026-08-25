/**
 * Projects: named creation, and what the rail remembers.
 *
 * `+` used to clear the selection and nothing else; the row was created by the
 * first submission. It now opens a name dialog and creates the project there and
 * then, which is the reversal recorded in `SessionList.tsx` and the reason two
 * things below have to be true rather than assumed:
 *
 *  - a project named on creation keeps that name once something is generated in
 *    it. The server back-fills a NULL title from the first prompt
 *    (`PlaygroundStore.add_generation`), and the fake server here models exactly
 *    that rule — back-fill only when the stored title is NULL — so the test
 *    fails if the interface creates the project and lets the prompt name it
 *    instead of sending the name first;
 *  - cancelling creates nothing, and a failed rename leaves a project the rail
 *    can show and delete rather than a record only the server knows about.
 *
 * The API keeps saying "session" throughout: `POST /playground/api/sessions`,
 * `PATCH …/{id}`. The rename is the interface's, not the route's.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { type FakeServer, fakeServer } from "../test-server";
import type { PlaygroundGeneration, PlaygroundSession } from "../types";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;

/** The store, as far as these tests are concerned: sessions and generations. */
let sessions: PlaygroundSession[];
let generations: PlaygroundGeneration[];

/**
 * The `title` of a `PATCH` body, narrowed rather than asserted.
 *
 * `Seen.body` is `unknown` on purpose — the fake server refuses to promise the
 * shape of what a component sent, which is the thing under test.
 */
function titleIn(body: unknown): string | null {
  if (body && typeof body === "object" && "title" in body && typeof body.title === "string") {
    return body.title;
  }
  return null;
}

function generation(sessionId: string, prompt: string): PlaygroundGeneration {
  return {
    id: "g1",
    sessionId,
    groupId: "g1",
    prompt,
    negativePrompt: null,
    rewrittenPrompt: null,
    rewriteError: null,
    model: "qwen-image-2512",
    kind: "txt2img",
    n: 1,
    size: "512x288",
    steps: 6,
    seeds: [7],
    contextImage: null,
    status: "completed",
    error: null,
    images: [{ url: "/playground/images/g1.png", seed: 7 }],
    createdAt: 2,
    startedAt: 2,
    finishedAt: 3,
  };
}

beforeEach(() => {
  sessions = [];
  generations = [];
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({ max_n: 4, default_model: "qwen-image-2512" }));
  server.on("GET /v1/progress", () => new Response(new ReadableStream({ start() {} })));
  server.on("GET /playground/api/upscalers", () => ({ upscalers: [] }));
  server.on("GET /playground/api/sessions", () => ({ sessions, paused: false }));

  // `create_session()` inserts with `title = NULL`.
  server.on("POST /playground/api/sessions", () => {
    const created: PlaygroundSession = {
      id: `s${sessions.length + 1}`,
      title: null,
      createdAt: 1,
      updatedAt: 1,
      generating: false,
      locked: false,
      // The server states it: a project born this second has nothing to show.
      cover: null,
    };
    sessions = [created, ...sessions];
    return created;
  });

  server.on("PATCH /playground/api/sessions/s1", ({ body }) => {
    const title = titleIn(body);
    sessions = sessions.map((entry) => (entry.id === "s1" ? { ...entry, title } : entry));
    return sessions.find((entry) => entry.id === "s1");
  });

  server.on("GET /playground/api/sessions/s1", () => ({
    session: sessions.find((entry) => entry.id === "s1"),
    generations,
  }));

  // The rule under test, on the server's side: the title is taken from the first
  // prompt *only while it is still NULL*. This is what `add_generation` does.
  server.on("POST /playground/api/sessions/s1/generations", ({ body }) => {
    const prompt = body instanceof FormData ? String(body.get("prompt")) : "";
    generations = [generation("s1", prompt)];
    sessions = sessions.map((entry) =>
      entry.id === "s1" ? { ...entry, title: entry.title ?? prompt, updatedAt: 3 } : entry,
    );
    return { id: "g1" };
  });

  window.history.replaceState(null, "", "/playground/");
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
  vi.restoreAllMocks();
});

async function createProject(user: UserEvent, name: string) {
  await user.click(await screen.findByRole("button", { name: "New project" }));
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  await user.type(within(dialog).getByLabelText("Name"), name);
  await user.click(within(dialog).getByRole("button", { name: "Create" }));
}

it("creates the project when it is named, and opens it", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await createProject(user, "Iceland");

  await waitFor(() => expect(screen.getByRole("button", { name: /^Iceland/ })).toBeTruthy());
  // Created, then named: the API has no create-with-title, so both calls are
  // made and in that order.
  const calls = server.requests.filter((request) =>
    request.path.startsWith("/playground/api/sessions"),
  );
  const create = calls.findIndex((request) => request.method === "POST");
  const name = calls.findIndex((request) => request.method === "PATCH");
  expect(create).toBeGreaterThanOrEqual(0);
  expect(name).toBeGreaterThan(create);
  expect(calls[name]!.body).toEqual({ title: "Iceland" });
  // Opened, so the studio shows the project that was just made.
  expect(new URLSearchParams(window.location.search).get("session")).toBe("s1");
});

it("keeps a named project's name after its first generation", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await createProject(user, "Iceland");
  await waitFor(() => expect(screen.getByRole("button", { name: /^Iceland/ })).toBeTruthy());

  // The prompt that would have named it, had it not been named already.
  await user.type(await screen.findByRole("textbox"), "a lone lighthouse in a storm");
  await user.click(screen.getByRole("button", { name: /generate/i }));

  await waitFor(() =>
    expect(screen.getByAltText("a lone lighthouse in a storm")).toBeTruthy(),
  );
  // The name the user gave it, not the prompt: the server only back-fills a
  // title that is still NULL, and this project's was set before it could run.
  // Asked of the rail, not of the page: the feed's own toolbars name the prompt
  // too, and "the prompt appears nowhere" is not the claim — "the rail still
  // says Iceland, and no row is named after the prompt" is.
  const rail = screen.getByRole("complementary");
  expect(within(rail).getByRole("button", { name: /^Iceland/ })).toBeTruthy();
  expect(within(rail).queryByRole("button", { name: /a lone lighthouse/ })).toBeNull();
  expect(sessions.find((entry) => entry.id === "s1")?.title).toBe("Iceland");
});

it("creates nothing when the name dialog is cancelled", async () => {
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await user.click(await screen.findByRole("button", { name: "New project" }));
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  await user.type(within(dialog).getByLabelText("Name"), "abandoned");
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

  // The dialog is what creates the project, so a cancel leaves the server
  // holding nothing — no empty project the user never confirmed.
  expect(
    server.requests.some(
      (request) => request.method === "POST" && request.path === "/playground/api/sessions",
    ),
  ).toBe(false);
  expect(sessions).toHaveLength(0);
});

it("leaves a project the rail can show and delete when naming it fails", async () => {
  const user = userEvent.setup();
  server.fail("PATCH /playground/api/sessions/s1", 500, "Could not rename it.");
  render(<PlaygroundApp />);
  await createProject(user, "Iceland");

  // The failure is reported in the dialog it happened in.
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  expect(within(dialog).getByRole("alert").textContent).toContain("Could not rename it.");

  // And the project the POST did create is already in the rail, named for what
  // it is, with its own Delete: the one outcome that had to be impossible is a
  // record the server holds and the interface cannot show or remove.
  expect(sessions).toHaveLength(1);
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /Untitled project/ })).toBeTruthy(),
  );
  expect(screen.getByRole("button", { name: "Delete project" })).toBeTruthy();
});

it("names the project the POST already made rather than making a second one", async () => {
  const user = userEvent.setup();
  server.fail("PATCH /playground/api/sessions/s1", 500, "Could not rename it.");
  render(<PlaygroundApp />);
  await createProject(user, "Iceland");
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  await within(dialog).findByRole("alert");

  // Retry, with the server willing this time.
  server.on("PATCH /playground/api/sessions/s1", ({ body }) => {
    const title = (body as { title: string | null }).title;
    sessions = sessions.map((entry) => (entry.id === "s1" ? { ...entry, title } : entry));
    return sessions.find((entry) => entry.id === "s1");
  });
  await user.click(within(dialog).getByRole("button", { name: "Create" }));

  await waitFor(() => expect(screen.getByRole("button", { name: /^Iceland/ })).toBeTruthy());
  // One project, not two: the retry named the record the first attempt created.
  expect(sessions).toHaveLength(1);
  expect(
    server.requests.filter(
      (request) => request.method === "POST" && request.path === "/playground/api/sessions",
    ),
  ).toHaveLength(1);
});

it("remembers the rail's collapsed state across mounts", async () => {
  // jsdom's `localStorage` is shadowed by Node's own under Node >= 22 and reads
  // back `undefined` (see `test-setup.ts`), so the store is installed here for
  // the duration of this test. It is a real `Storage`-shaped seam: what is being
  // witnessed is that the rail writes its state and reads it back on the next
  // mount, not that a browser has storage.
  const store = new Map<string, string>();
  const stub = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
    key: () => null,
    length: 0,
  };
  const original = Object.getOwnPropertyDescriptor(window, "localStorage");
  Object.defineProperty(window, "localStorage", { value: stub, configurable: true });
  try {
    const user = userEvent.setup();
    const first = render(<PlaygroundApp />);
    await user.click(
      await screen.findByRole("button", { name: "Collapse the project rail" }),
    );
    expect(store.get("qds.playground.rail-collapsed")).toBe("1");
    first.unmount();

    render(<PlaygroundApp />);
    // Collapsed on the way back in: the control offers to expand it.
    expect(
      await screen.findByRole("button", { name: "Expand the project rail" }),
    ).toBeTruthy();
  } finally {
    if (original) Object.defineProperty(window, "localStorage", original);
    else Reflect.deleteProperty(window, "localStorage");
  }
});
