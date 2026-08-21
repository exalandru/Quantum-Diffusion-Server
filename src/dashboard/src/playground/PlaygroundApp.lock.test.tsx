/**
 * A locked session, from the studio's point of view.
 *
 * The server answers 403 `session_locked`; what must follow is an unlock prompt
 * — not the admin login a 401 opens, and not the "could not reach the server"
 * banner — and, once the password is taken, every request for that session
 * carrying the token it was exchanged for, images included. Renaming rides
 * along: same route family, same refresh.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
  prompt: "a fox",
  model: "qwen-image-2512",
  kind: "txt2img",
  n: 1,
  size: "512x288",
  steps: 6,
  seeds: [41],
  contextImage: "/playground/images/ctx-1.png",
  status: "completed",
  error: null,
  images: [{ url: "/playground/images/g1.png", seed: 41 }],
  createdAt: 1,
  startedAt: 1,
  finishedAt: 2,
};

const TOKEN = "tok-123";
const LOCKED_HEADER = "x-qds-session-token";

beforeEach(() => {
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({
    max_n: 4,
    default_model: "qwen-image-2512",
  }));
  // A stream that never yields: the panel stays subscribed and idle.
  server.on(
    "GET /v1/progress",
    () => new Response(new ReadableStream({ start() {} })),
  );
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "foxes", locked: true })],
  }));
  // Locked until the header carries the token.
  server.on("GET /playground/api/sessions/s1", ({ headers }) =>
    headers[LOCKED_HEADER] === TOKEN
      ? {
          session: playgroundSession({
            id: "s1",
            title: "foxes",
            locked: true,
          }),
          generations: [GENERATION],
        }
      : new Response(
          JSON.stringify({
            error: {
              message: "Playground session 's1' is locked.",
              type: "invalid_request_error",
              param: null,
              code: "session_locked",
            },
          }),
          { status: 403, headers: { "Content-Type": "application/json" } },
        ),
  );
  window.history.replaceState(null, "", "/playground/?session=s1");
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
});

async function openLockedSession() {
  render(<PlaygroundApp />);
  const dialog = await screen.findByRole("dialog", { name: "Unlock session" });
  return dialog;
}

it("asks for the session password, not the admin one, and does not call the server down", async () => {
  await openLockedSession();
  expect(screen.queryByText(/admin password/i)).toBeNull();
  expect(screen.queryByText(/could not reach the server/i)).toBeNull();
  expect(
    screen.getByRole("heading", { name: "This session is locked" }),
  ).toBeTruthy();
});

it("keeps the prompt open on a wrong password, with the server's words", async () => {
  server.fail(
    "POST /playground/api/sessions/s1/unlock",
    403,
    "Wrong session password.",
    "invalid_session_password",
  );
  const user = userEvent.setup();
  const dialog = await openLockedSession();
  await user.type(within(dialog).getByLabelText("Password"), "nope-nope");
  await user.click(within(dialog).getByRole("button", { name: "Unlock" }));
  expect((await within(dialog).findByRole("alert")).textContent).toBe(
    "Wrong session password.",
  );
  expect(screen.getByRole("dialog", { name: "Unlock session" })).toBeTruthy();
});

it("exchanges the password for a token and sends it from then on, images included", async () => {
  server.on("POST /playground/api/sessions/s1/unlock", ({ body }) => {
    expect(body).toEqual({ password: "correct horse battery" });
    return {
      token: TOKEN,
      session: playgroundSession({ id: "s1", title: "foxes", locked: true }),
    };
  });
  const user = userEvent.setup();
  const dialog = await openLockedSession();
  await user.type(
    within(dialog).getByLabelText("Password"),
    "correct horse battery",
  );
  await user.click(within(dialog).getByRole("button", { name: "Unlock" }));

  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(window.sessionStorage.getItem("qds.playground.unlock.s1")).toBe(TOKEN);

  // The feed loaded with the token, and its images carry it on the URL —
  // the one place the server takes it as a query, because `<img>` cannot
  // send a header.
  const image = await screen.findByRole("img", { name: "a fox" });
  expect(image.getAttribute("src")).toBe(
    `/playground/images/g1.png?t=${TOKEN}`,
  );
  expect(
    screen.getByRole("img", { name: "Reference image" }).getAttribute("src"),
  ).toBe(`/playground/images/ctx-1.png?t=${TOKEN}`);
  const detail = server.requests.filter(
    (r) => r.method === "GET" && r.path === "/playground/api/sessions/s1",
  );
  expect(detail.at(-1)?.headers[LOCKED_HEADER]).toBe(TOKEN);
  // And the sidebar says so.
  expect(
    screen.getByRole("img", { name: "Locked, open in this tab" }),
  ).toBeTruthy();
});

it("locking forgets the token and shows the locked studio again", async () => {
  window.sessionStorage.setItem("qds.playground.unlock.s1", TOKEN);
  server.on("POST /playground/api/sessions/s1/lock", ({ headers }) => {
    expect(headers[LOCKED_HEADER]).toBe(TOKEN);
    return new Response(null, { status: 204 });
  });
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findByRole("img", { name: "a fox" });

  await user.click(screen.getByRole("button", { name: "Lock foxes" }));
  await screen.findByRole("heading", { name: "This session is locked" });
  expect(window.sessionStorage.getItem("qds.playground.unlock.s1")).toBeNull();
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("a stale token is dropped and the prompt opens once", async () => {
  window.sessionStorage.setItem("qds.playground.unlock.s1", "expired");
  await openLockedSession();
  expect(window.sessionStorage.getItem("qds.playground.unlock.s1")).toBeNull();
  expect(screen.getAllByRole("dialog")).toHaveLength(1);
});

it("renames through PATCH and refreshes the list", async () => {
  window.history.replaceState(null, "", "/playground/");
  let title = "foxes";
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title })],
  }));
  server.on("PATCH /playground/api/sessions/s1", ({ body }) => {
    title = (body as { title: string }).title;
    return playgroundSession({ id: "s1", title });
  });
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await user.click(await screen.findByRole("button", { name: "Rename foxes" }));
  const dialog = screen.getByRole("dialog", { name: "Rename session" });
  const field = within(dialog).getByLabelText("Name");
  expect((field as HTMLInputElement).value).toBe("foxes");
  await user.clear(field);
  await user.type(field, "Vulpes");
  await user.click(within(dialog).getByRole("button", { name: "Rename" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(await screen.findByText("Vulpes")).toBeTruthy();
  expect(server.requests.find((r) => r.method === "PATCH")?.body).toEqual({
    title: "Vulpes",
  });
});

it("setting a password keeps this tab unlocked with the returned token", async () => {
  window.history.replaceState(null, "", "/playground/");
  let locked = false;
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "foxes", locked })],
  }));
  server.on("POST /playground/api/sessions/s1/password", ({ body }) => {
    expect(body).toEqual({ password: "correct horse battery" });
    locked = true;
    return { token: TOKEN };
  });
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await user.click(
    await screen.findByRole("button", { name: "Set a password on foxes" }),
  );
  const dialog = screen.getByRole("dialog", { name: "Set a password" });
  await user.type(
    within(dialog).getByLabelText(/^Password/),
    "correct horse battery",
  );
  await user.type(
    within(dialog).getByLabelText("Confirm"),
    "correct horse battery",
  );
  await user.click(
    within(dialog).getByRole("button", { name: "Set password" }),
  );
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(window.sessionStorage.getItem("qds.playground.unlock.s1")).toBe(TOKEN);
  expect(
    await screen.findByRole("img", { name: "Locked, open in this tab" }),
  ).toBeTruthy();
});

it("unlocks before deleting a locked session, then deletes with the token", async () => {
  window.history.replaceState(null, "", "/playground/");
  let deleted = false;
  server.on("DELETE /playground/api/sessions/s1", ({ headers }) => {
    if (headers[LOCKED_HEADER] !== TOKEN)
      return new Response(
        JSON.stringify({
          error: {
            message: "locked",
            type: "invalid_request_error",
            param: null,
            code: "session_locked",
          },
        }),
        { status: 403, headers: { "Content-Type": "application/json" } },
      );
    deleted = true;
    return new Response(null, { status: 204 });
  });
  server.on("POST /playground/api/sessions/s1/unlock", () => ({
    token: TOKEN,
    session: playgroundSession({ id: "s1", locked: true }),
  }));
  vi.spyOn(window, "confirm").mockReturnValue(true);
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await user.click(await screen.findByRole("button", { name: "Delete foxes" }));
  const dialog = await screen.findByRole("dialog", { name: "Unlock session" });
  await user.type(
    within(dialog).getByLabelText("Password"),
    "correct horse battery",
  );
  await user.click(within(dialog).getByRole("button", { name: "Unlock" }));
  await waitFor(() => expect(deleted).toBe(true));
  await act(async () => {});
});

it("refining fetches the source image with the token on the URL", async () => {
  window.sessionStorage.setItem("qds.playground.unlock.s1", TOKEN);
  server.on(
    `GET /playground/images/g1.png?t=${TOKEN}`,
    () => new Response(new Blob([new Uint8Array([1])])),
  );
  server.on("POST /playground/api/sessions/s1/generations", ({ headers }) => {
    expect(headers[LOCKED_HEADER]).toBe(TOKEN);
    return { ...GENERATION, id: "g2", status: "queued", images: [] };
  });
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findByRole("img", { name: "a fox" });
  await user.click(screen.getByRole("button", { name: /refine/i }));
  await waitFor(() =>
    expect(
      server.requests.some(
        (r) => r.method === "POST" && r.path.endsWith("/generations"),
      ),
    ).toBe(true),
  );
  expect(
    server.requests.some((r) => r.path === "/playground/images/g1.png"),
  ).toBe(true);
});
