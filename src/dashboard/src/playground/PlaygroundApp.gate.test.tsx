/**
 * The playground's own gate, from the studio's point of view.
 *
 * Three refusals meet this surface and they are three different things, which is
 * the whole point of this file:
 *
 * * 403 `session_locked` — one project inside an open playground. That is
 *   `PlaygroundApp.lock.test.tsx`, and nothing here may disturb it.
 * * 401 with the plane gated — this plane wants *its* password, and the form
 *   that renders must not be the admin one: typing the admin password into a
 *   screen that only needs the playground's is handing over the configuration
 *   writer for no reason.
 * * 401 with the plane not gated — a server with an `api_key` and no playground
 *   password, which is how it has always behaved. The admin login opens it,
 *   because an admin session opens this plane too, and that path must keep
 *   working.
 *
 * `gated` comes from the server rather than from the status code, because the
 * status code cannot carry it: both cases are 401.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { type FakeServer, fakeServer } from "../test-server";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;

/** Everything the studio needs once it is through the gate. */
function serveTheStudio() {
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({ max_n: 4, default_model: "qwen-image-2512" }));
  server.on("GET /v1/progress", () => new Response(new ReadableStream({ start() {} })));
  server.on("GET /playground/api/upscalers", () => ({ upscalers: [] }));
}

/** The plane refusing everything, the way a gated server does. */
function refuseTheApi() {
  server.fail(
    "GET /playground/api/sessions",
    401,
    "This server's playground requires the playground password.",
    "playground_login_required",
  );
}

beforeEach(() => {
  server = fakeServer();
  serveTheStudio();
  refuseTheApi();
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
});

it("asks for the playground password when the plane is gated, not the admin one", async () => {
  server.on("GET /playground/api/session", () => ({
    passwordSet: true,
    authenticated: false,
    loopback: true,
    gated: true,
  }));

  render(<PlaygroundApp />);

  expect(await screen.findByRole("heading", { name: "Playground password" })).toBeTruthy();
  expect(screen.getByLabelText("Password")).toBeTruthy();
  // The negative that gives the positive its meaning: the admin form is what
  // this surface used to render for every 401.
  expect(screen.queryByRole("heading", { name: "Admin password" })).toBeNull();
  expect(screen.queryByText(/could not reach the server/i)).toBeNull();
  // And nothing here asked the control plane anything.
  expect(server.requests.some((seen) => seen.path === "/admin/session")).toBe(false);
});

it("takes the password, then loads the studio it was refused", async () => {
  server.on("GET /playground/api/session", () => ({
    passwordSet: true,
    authenticated: false,
    loopback: true,
    gated: true,
  }));
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findByRole("heading", { name: "Playground password" });

  // The login succeeds and the plane opens: the same route that was refusing
  // now answers, which is what a cookie the server accepted looks like here.
  server.on("POST /playground/api/session", ({ body }) => {
    expect(body).toEqual({ password: "open the picture door" });
    server.on("GET /playground/api/sessions", () => ({
      sessions: [playgroundSession({ id: "s1", title: "foxes" })],
      paused: false,
    }));
    return new Response(null, { status: 204 });
  });

  await user.type(screen.getByLabelText("Password"), "open the picture door");
  await user.click(screen.getByRole("button", { name: "Unlock" }));

  // Re-fetched rather than merely dismissed: a lock screen that closes onto an
  // empty studio would be a screen that lied about what it unlocked.
  expect(await screen.findByText("foxes")).toBeTruthy();
  expect(screen.queryByRole("heading", { name: "Playground password" })).toBeNull();
});

it("keeps the form up on a wrong password, with the server's words", async () => {
  server.on("GET /playground/api/session", () => ({
    passwordSet: true,
    authenticated: false,
    loopback: true,
    gated: true,
  }));
  server.fail("POST /playground/api/session", 401, "Incorrect password.", "invalid_password");
  const user = userEvent.setup();
  render(<PlaygroundApp />);
  await screen.findByRole("heading", { name: "Playground password" });

  await user.type(screen.getByLabelText("Password"), "not-it-either");
  await user.click(screen.getByRole("button", { name: "Unlock" }));

  expect((await screen.findByRole("status")).textContent).toBe("Incorrect password.");
  expect(screen.getByLabelText("Password")).toBeTruthy();
});

it("says so rather than offering a form when the gate has no password behind it", async () => {
  server.on("GET /playground/api/session", () => ({
    passwordSet: false,
    authenticated: false,
    loopback: true,
    gated: true,
  }));

  render(<PlaygroundApp />);

  expect(await screen.findByRole("heading", { name: "Playground password" })).toBeTruthy();
  // A form with nothing to check against is a dead end that looks like a way in.
  expect(screen.queryByLabelText("Password")).toBeNull();
  expect(screen.getByText(/none is set/i)).toBeTruthy();
});

it("still sends an api_key-only server to the admin login", async () => {
  // The plane is not gated: this is the refusal a configured `api_key` produces,
  // and an admin session is what opens it. Regression guard for the surface the
  // Hermes plugin and Open WebUI use.
  server.on("GET /playground/api/session", () => ({
    passwordSet: false,
    authenticated: false,
    loopback: true,
    gated: false,
  }));
  server.on("GET /admin/session", () => ({
    passwordSet: true,
    authenticated: false,
    loopback: true,
    recoveryMode: false,
  }));

  render(<PlaygroundApp />);

  expect(await screen.findByRole("heading", { name: "Admin password" })).toBeTruthy();
  expect(screen.queryByRole("heading", { name: "Playground password" })).toBeNull();
});

it("renders the compact form in an embedder's pane", async () => {
  // `?view=plugin` is where the lock is most likely to be met: the plugin's pane
  // is a fresh browsing context holding no cookie.
  window.history.replaceState(null, "", "/playground/?view=plugin");
  server.on("GET /playground/api/session", () => ({
    passwordSet: true,
    authenticated: false,
    loopback: true,
    gated: true,
  }));

  render(<PlaygroundApp />);

  const heading = await screen.findByRole("heading", { name: "Playground password" });
  const card = heading.closest("section") as HTMLElement;
  expect(card.className).toContain("login-card-compact");
  // No page masthead in a pane that is not a page.
  expect(screen.queryByRole("heading", { name: "Playground", level: 1 })).toBeNull();
  // Still a form: the compact variant is a size, not a lesser surface.
  expect(screen.getByLabelText("Password")).toBeTruthy();
});

it("does not ask for a password at all when the plane is open", async () => {
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "foxes" })],
    paused: false,
  }));
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [],
  }));

  render(<PlaygroundApp />);

  expect(await screen.findByText("foxes")).toBeTruthy();
  await waitFor(() =>
    expect(server.requests.some((seen) => seen.path === "/playground/api/session")).toBe(false),
  );
});
