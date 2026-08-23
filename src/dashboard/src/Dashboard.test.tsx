/**
 * What the Dashboard may claim when it has no reading, and what it says about a
 * progress stream that has stopped.
 *
 * The first was found in release QA: with no `/health` answer the runtime panel
 * printed "keep warm" for the release policy — the opposite of the configured
 * 10-second release. An absent reading is not a value.
 *
 * The second is the live stream. The panel no longer receives an injected
 * client: it subscribes through `api.subscribeProgress`, which is `fetch` to
 * `/v1/progress` and owns its own bounded retry. So these tests drive the real
 * transport — a real SSE body, framed as the server frames it — rather than a
 * stand-in for it, and the reconnect they observe is the one that ships.
 */
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { Dashboard } from "./panels/Dashboard";
import type { JobView } from "./job";
import { job, overview } from "./test-fixtures";
import { type FakeServer, fakeServer } from "./test-server";
import type { Health, Progress } from "./types";

let server: FakeServer;

function idleJobs(): JobView {
  return {
    job: job(),
    error: null,
    active: false,
    refresh: vi.fn(async () => {}),
    onSettled: () => () => {},
  };
}

function health(patch: Partial<Health> = {}): Health {
  return {
    status: "ok",
    version: "1.0.0",
    default_model: "z-image",
    models: ["z-image"],
    loaded_model: null,
    idle_unload_s: null,
    memory: {},
    ...patch,
  };
}

function frame(patch: Partial<Progress> = {}): string {
  const progress: Progress = {
    state: "generating",
    model: "z-image",
    kind: "txt2img",
    seed: 1,
    step: 1,
    total: 9,
    preview_seq: 0,
    elapsed_s: 0.5,
    loaded_model: "z-image",
    memory: {},
    ...patch,
  };
  return `data: ${JSON.stringify(progress)}\n\n`;
}

/**
 * The `/v1/progress` endpoint, as a body the test writes into.
 *
 * `opens` is the count of connections the *transport* made, which is what makes
 * a reconnect distinguishable from a re-subscription: the panel subscribes once
 * and `subscribeProgress` is what dials again.
 */
function progressRoute() {
  const state = { opens: 0 };
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  const encoder = new TextEncoder();

  server.on("GET /v1/progress", () => {
    state.opens += 1;
    return new Response(
      new ReadableStream<Uint8Array>({
        start(c) {
          controller = c;
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });

  return {
    state,
    /** Push one SSE frame down the currently-open body. */
    emit: (patch: Partial<Progress> = {}) => controller?.enqueue(encoder.encode(frame(patch))),
    /** Break the connection the way a lost network does. */
    drop: (message: string) => controller?.error(new Error(message)),
  };
}

/** The tile's value, found through its own label rather than by position. */
function stat(label: string): string {
  const dt = screen.getByText(label);
  const tile = dt.closest("div");
  if (!tile) throw new Error(`no tile for ${label}`);
  return within(tile).getAllByText((_, node) => node?.tagName === "DD")[0]!.textContent ?? "";
}

beforeEach(() => {
  server = fakeServer();
});
afterEach(() => {
  server.restore();
  vi.restoreAllMocks();
});

it("does not claim a release policy it has not been told", async () => {
  // No `/health` answer at all: the shell's poll has nothing either, so the
  // panel is mounted with `health={null}` and must say it does not know.
  server.fail("GET /health", 503, "the engine is not answering");
  progressRoute();

  render(<Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />);

  await screen.findByText("Release policy");
  expect(stat("Release policy")).toBe("-");
  expect(screen.queryByText("keep warm")).toBeNull();
});

it("reports each policy the server actually declares", async () => {
  const cases: [number | null, string][] = [
    [null, "keep warm"],
    [0, "after each request"],
    [10, "after 10s idle"],
  ];
  for (const [value, expected] of cases) {
    server.on("GET /health", () => health({ idle_unload_s: value }));
    progressRoute();

    const view = render(
      <Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />,
    );
    await screen.findByText(expected);
    expect(stat("Release policy")).toBe(expected);
    view.unmount();
  }
});

// ── The progress stream, as the panel sees it ──────────────────────────────

it("reports a dropped stream and clears it when events resume, with no navigation", async () => {
  server.on("GET /health", () => health({ idle_unload_s: 10 }));
  const stream = progressRoute();

  render(<Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />);

  const activity = await screen.findByText("Current activity");
  await waitFor(() => expect(stream.state.opens).toBe(1));

  await act(async () => stream.drop("network error"));
  expect(await screen.findByText("reconnecting")).toBeTruthy();
  expect(screen.getByText(/Retrying automatically/)).toBeTruthy();

  // The transport reconnects itself — a second connection, from one
  // subscription — and the first frame of the recovered stream clears the note.
  await waitFor(() => expect(stream.state.opens).toBe(2), { timeout: 3000 });
  await act(async () => {
    stream.emit({ step: 4 });
  });

  await waitFor(() => expect(screen.queryByText("reconnecting")).toBeNull());
  expect(screen.getByText(/step 4\/9/)).toBeTruthy();
  // The panel was never remounted and no tab was touched.
  expect(activity.isConnected).toBe(true);
});

it("stops retrying when the panel goes away, rather than dialling forever", async () => {
  // The retry loop cannot outlive the thing that owns it. Losing the connection
  // is what proves the loop is running at all; unmounting is what must end it.
  server.on("GET /health", () => health({ idle_unload_s: 10 }));
  const stream = progressRoute();

  const view = render(
    <Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />,
  );
  await screen.findByText("Release policy");
  await waitFor(() => expect(stream.state.opens).toBe(1));

  await act(async () => stream.drop("network error"));
  await waitFor(() => expect(stream.state.opens).toBe(2), { timeout: 3000 });

  view.unmount();
  const afterUnmount = stream.state.opens;

  // Watch for a further connection for long enough to cover several backoffs
  // (`RECONNECT_BASE_MS` is 500ms). `waitFor` rejecting is the pass: it means
  // the count never grew, so the loop really did stop.
  let dialledAgain = false;
  try {
    await waitFor(() => expect(stream.state.opens).toBeGreaterThan(afterUnmount), {
      timeout: 1200,
    });
    dialledAgain = true;
  } catch {
    // Expected: no reconnection outlived the panel.
  }
  expect(dialledAgain).toBe(false);
});

it("keeps an action's result independent of the stream's state", async () => {
  // Slice 1's separation, checked against the new channel: a reconnect notice
  // and a failed action are different things and must not overwrite each other.
  server.on("GET /health", () => health({ idle_unload_s: 10 }));
  server.fail("POST /admin/restart", 500, "the server did not restart in time");
  const stream = progressRoute();

  render(<Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />);
  await screen.findByText("Release policy");
  await waitFor(() => expect(stream.state.opens).toBe(1));

  await userEvent.click(screen.getByRole("button", { name: "Restart server" }));
  await screen.findByText("the server did not restart in time");

  await act(async () => stream.drop("network error"));
  expect(await screen.findByText("reconnecting")).toBeTruthy();
  expect(screen.getByText("the server did not restart in time")).toBeTruthy();

  // And the stream recovering does not clear the action's result either.
  await waitFor(() => expect(stream.state.opens).toBe(2), { timeout: 3000 });
  await act(async () => {
    stream.emit({ step: 2 });
  });
  await waitFor(() => expect(screen.queryByText("reconnecting")).toBeNull());
  expect(screen.getByText("the server did not restart in time")).toBeTruthy();
});

// ── Lifecycle feedback ─────────────────────────────────────────────────────

it("keeps a failed action on screen while a different one succeeds", async () => {
  // Each button owns its own slot, so a later success is not evidence about an
  // earlier failure: the refusal survives until that button runs again or it is
  // dismissed, and nothing a poll does may clear it.
  server.on("GET /health", () => health({ idle_unload_s: 10 }));
  server.fail("POST /admin/restart", 500, "the server did not restart in time");
  server.on("POST /v1/unload", () => ({ loaded_model: null, memory: {} }));
  const stream = progressRoute();

  render(<Dashboard state={overview()} health={null} jobs={idleJobs()} onChanged={() => {}} />);
  await screen.findByText("Release policy");
  await waitFor(() => expect(stream.state.opens).toBe(1));
  // Something has to be loaded for "Free memory" to be offered at all.
  await act(async () => {
    stream.emit({ state: "idle", step: 0, total: 0, loaded_model: "z-image" });
  });

  await userEvent.click(screen.getByRole("button", { name: "Restart server" }));
  await screen.findByText("the server did not restart in time");

  await userEvent.click(screen.getByRole("button", { name: "Free memory" }));
  await screen.findByText("Model released.");

  // Both, at once, and distinguishable: the failure was not overwritten.
  expect(screen.getByText("the server did not restart in time")).toBeTruthy();

  // Dismissing is the user's, and it is the only thing that clears it.
  const failure = screen.getByText("the server did not restart in time").closest("div")!;
  await userEvent.click(within(failure).getByRole("button", { name: "Dismiss" }));
  expect(screen.queryByText("the server did not restart in time")).toBeNull();
});
