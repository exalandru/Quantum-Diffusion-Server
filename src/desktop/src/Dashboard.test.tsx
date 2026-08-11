/**
 * What the Dashboard may claim when it has no reading.
 *
 * Found in release QA: with the server stopped there is no `/health`, and the
 * runtime panel printed "keep warm" for the release policy — the opposite of the
 * configured 10-second release. An absent reading is not a value.
 */
import { invoke } from "@tauri-apps/api/core";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { Dashboard } from "./panels/Dashboard";
import type { JobView } from "./job";
import { job, overview } from "./test-fixtures";
import type { Health } from "./types";

const mockInvoke = vi.mocked(invoke);

function idleJobs(): JobView {
  return { job: job(), error: null, active: false, refresh: vi.fn(async () => {}), onSettled: () => () => {} };
}

function health(patch: Partial<Health> = {}): Health {
  return {
    status: "ok",
    version: "0.2.0",
    default_model: "z-image",
    models: ["z-image"],
    loaded_model: null,
    idle_unload_s: null,
    memory: {},
    ...patch,
  };
}

/** The tile's value, found through its own label rather than by position. */
function stat(label: string): string {
  const dt = screen.getByText(label);
  const tile = dt.closest("div");
  if (!tile) throw new Error(`no tile for ${label}`);
  return within(tile).getAllByText((_, node) => node?.tagName === "DD")[0]!.textContent ?? "";
}

beforeEach(() => mockInvoke.mockReset());
afterEach(() => vi.restoreAllMocks());

it("does not claim a release policy it has not been told", async () => {
  // Server stopped: `client` is null, so nothing has answered `/health`.
  render(<Dashboard state={overview()} client={null} jobs={idleJobs()} onChanged={() => {}} />);
  expect(stat("Release policy")).toBe("—");
  expect(screen.queryByText("keep warm")).toBeNull();
});

it("reports each policy the server actually declares", async () => {
  const cases: [number | null, string][] = [
    [null, "keep warm"],
    [0, "after each request"],
    [10, "after 10s idle"],
  ];
  for (const [value, expected] of cases) {
    const client = {
      health: async () => health({ idle_unload_s: value }),
      subscribeProgress: () => () => {},
    } as never;
    const view = render(
      <Dashboard
        state={overview({ server: { running: true, port: 8765, lastExit: null } })}
        client={client}
        jobs={idleJobs()}
        onChanged={() => {}}
      />,
    );
    await screen.findByText(expected);
    expect(stat("Release policy")).toBe(expected);
    view.unmount();
  }
});

// ── The progress stream, as the panel sees it ──────────────────────────────

/** A client whose stream the test drives, standing in for `ServerClient`. */
function streamingClient() {
  let onProgress: (p: unknown) => void = () => {};
  let onError: (m: string) => void = () => {};
  const state = { subscriptions: 0, live: 0 };
  const client = {
    health: async () => health({ idle_unload_s: 10 }),
    subscribeProgress: (progress: (p: unknown) => void, error: (m: string) => void) => {
      state.subscriptions += 1;
      state.live += 1;
      onProgress = progress;
      onError = error;
      return () => {
        state.live -= 1;
      };
    },
  } as never;
  return {
    client,
    state,
    drop: (message: string) => onError(message),
    emit: (p: Record<string, unknown>) =>
      onProgress({
        state: "generating",
        model: "z-image",
        kind: "txt2img",
        seed: 1,
        step: 1,
        total: 9,
        elapsed_s: 0.5,
        loaded_model: "z-image",
        memory: {},
        ...p,
      }),
  };
}

it("reports a dropped stream and clears it when events resume, with no navigation", async () => {
  const { client, state, drop, emit } = streamingClient();
  const running = overview({ server: { running: true, port: 8765, lastExit: null } });
  render(<Dashboard state={running} client={client} jobs={idleJobs()} onChanged={() => {}} />);

  await screen.findByText("Release policy");
  expect(state.subscriptions).toBe(1);

  await act(async () => drop("network error"));
  expect(screen.getByText("reconnecting")).toBeTruthy();
  expect(screen.getByText(/Retrying automatically/)).toBeTruthy();

  // The transport reconnects itself; the first frame of the recovered stream is
  // what clears the note. The panel is never remounted and no tab is touched.
  await act(async () => emit({ step: 4 }));
  expect(screen.queryByText("reconnecting")).toBeNull();
  expect(state.subscriptions).toBe(1);
  expect(screen.getByText(/step 4\/9/)).toBeTruthy();
});

it("unsubscribes exactly once when the server stops, rather than retrying forever", async () => {
  // Stopping the server deliberately makes `client` null upstream, which tears
  // the subscription down: the retry loop cannot outlive the thing that owns it.
  const { client, state } = streamingClient();
  const running = overview({ server: { running: true, port: 8765, lastExit: null } });
  const view = render(
    <Dashboard state={running} client={client} jobs={idleJobs()} onChanged={() => {}} />,
  );
  await screen.findByText("Release policy");
  expect(state.live).toBe(1);

  view.rerender(
    <Dashboard state={overview()} client={null} jobs={idleJobs()} onChanged={() => {}} />,
  );
  expect(state.live).toBe(0);
  expect(state.subscriptions).toBe(1);
});

it("tears the stream down on unmount", async () => {
  const { client, state } = streamingClient();
  const running = overview({ server: { running: true, port: 8765, lastExit: null } });
  const view = render(
    <Dashboard state={running} client={client} jobs={idleJobs()} onChanged={() => {}} />,
  );
  await screen.findByText("Release policy");
  view.unmount();
  expect(state.live).toBe(0);
});

it("keeps an action's result independent of the stream's state", async () => {
  // Slice 1's separation, checked against the new channel: a reconnect notice
  // and a failed action are different things and must not overwrite each other.
  mockInvoke.mockImplementation(async (command: string) => {
    if (command === "server_stop") throw new Error("the server did not stop in time");
    throw new Error(`unexpected ${command}`);
  });
  const { client, drop, emit } = streamingClient();
  const running = overview({ server: { running: true, port: 8765, lastExit: null } });
  render(<Dashboard state={running} client={client} jobs={idleJobs()} onChanged={() => {}} />);
  await screen.findByText("Release policy");

  await userEvent.click(screen.getByRole("button", { name: "Stop" }));
  await screen.findByText("the server did not stop in time");

  await act(async () => drop("network error"));
  expect(screen.getByText("reconnecting")).toBeTruthy();
  expect(screen.getByText("the server did not stop in time")).toBeTruthy();

  // And the stream recovering does not clear the action's result either.
  await act(async () => emit({ step: 2 }));
  expect(screen.queryByText("reconnecting")).toBeNull();
  expect(screen.getByText("the server did not stop in time")).toBeTruthy();
});
