/**
 * Holding the queue, and deleting a whole entry, from the studio's point of view.
 *
 * The pause is the server's state, not this page's: the button asks, and the
 * session list is what the answer is read back from. What is asserted here is
 * that the page says so — a held queue that looks identical to a working one is
 * the failure this feature exists to prevent.
 */

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { type FakeServer, fakeServer } from "../test-server";
import type { PlaygroundGeneration } from "../types";
import { PlaygroundApp } from "./PlaygroundApp";

Element.prototype.scrollIntoView = () => {};

let server: FakeServer;
let paused = false;

const GENERATION: PlaygroundGeneration = {
  id: "g1",
  sessionId: "s1",
  groupId: "lineage-1",
  prompt: "a fox",
  negativePrompt: null,
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
  paused = false;
  server = fakeServer();
  server.on("GET /v1/models", () => ({ object: "list", data: [] }));
  server.on("GET /v1/capabilities", () => ({ max_n: 4, default_model: "qwen-image-2512" }));
  server.on("GET /v1/progress", () => new Response(new ReadableStream({ start() {} })));
  // The pause is server state: the list reports whatever the last POST set.
  server.on("GET /playground/api/sessions", () => ({
    sessions: [playgroundSession({ id: "s1", title: "foxes" })],
    paused,
  }));
  server.on("POST /playground/api/queue", ({ body }) => {
    paused = (body as { paused: boolean }).paused;
    return { paused };
  });
  server.on("GET /playground/api/sessions/s1", () => ({
    session: playgroundSession({ id: "s1", title: "foxes" }),
    generations: [GENERATION],
  }));
  window.history.replaceState(null, "", "/playground/?session=s1");
});

afterEach(() => {
  server.restore();
  window.history.replaceState(null, "", "/playground/");
});

it("asks the server to hold the queue, and says so once it has", async () => {
  render(<PlaygroundApp />);
  const button = await screen.findByRole("button", { name: "Pause queue" });
  expect(screen.getByText("Running")).toBeTruthy();

  await userEvent.click(button);

  await waitFor(() => {
    expect(server.requests.find((entry) => entry.path === "/playground/api/queue")?.body).toEqual({
      paused: true,
    });
  });
  // The page says it in three places, because a held queue that looks like a
  // working one is exactly the confusion this is for.
  await screen.findByRole("button", { name: "Resume queue" });
  expect(screen.getByText("Queue paused")).toBeTruthy();
  expect(screen.getByText(/will finish, then nothing more starts/)).toBeTruthy();
  await act(async () => {});
});

it("releases the queue again", async () => {
  paused = true;
  render(<PlaygroundApp />);

  await userEvent.click(await screen.findByRole("button", { name: "Resume queue" }));

  await waitFor(() => {
    const posts = server.requests.filter((entry) => entry.path === "/playground/api/queue");
    expect(posts.at(-1)?.body).toEqual({ paused: false });
  });
  await screen.findByRole("button", { name: "Pause queue" });
  expect(screen.queryByText(/nothing more starts/)).toBeNull();
  await act(async () => {});
});

it("reports a queue another tab held, without being told by the button", async () => {
  // The state is global and this tab did not set it: it has to come off the
  // list, or a second tab submits into a held queue with no explanation.
  paused = true;
  render(<PlaygroundApp />);

  expect(await screen.findByText("Queue paused")).toBeTruthy();
  await act(async () => {});
});

it("puts the button back when the server refuses the hold", async () => {
  server.fail("POST /playground/api/queue", 500, "The runner is not running.");
  render(<PlaygroundApp />);

  await userEvent.click(await screen.findByRole("button", { name: "Pause queue" }));

  // Not left claiming a hold that never took: the optimistic state is undone.
  await screen.findByRole("button", { name: "Pause queue" });
  expect(screen.queryByText("Queue paused")).toBeNull();
  await act(async () => {});
});

it("deletes a whole entry by its group, and refreshes both views", async () => {
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  server.on("DELETE /playground/api/groups/lineage-1", () => new Response(null, { status: 204 }));
  render(<PlaygroundApp />);

  await userEvent.click(await screen.findByRole("button", { name: "Delete entry: a fox" }));

  await waitFor(() => {
    expect(
      server.requests.some(
        (entry) =>
          entry.method === "DELETE" && entry.path === "/playground/api/groups/lineage-1",
      ),
    ).toBe(true);
  });
  // Both, not just the transcript: the server bumps `updated_at`, which is the
  // sidebar's sort key, and an idle session polls nothing.
  const after = server.requests.slice(
    server.requests.findIndex((entry) => entry.method === "DELETE"),
  );
  expect(after.some((entry) => entry.path === "/playground/api/sessions/s1")).toBe(true);
  expect(after.some((entry) => entry.path === "/playground/api/sessions")).toBe(true);
  confirm.mockRestore();
  await act(async () => {});
});
