/**
 * The shell: what survives navigation, what survives a failing poll, and what
 * the Setup gate says.
 *
 * These three are the invariants the redesign was most able to break, because
 * all three are about state *outliving* a render — an action result outliving a
 * background poll (Slice 1), a Rust-owned job outliving a view switch (Slice 2),
 * and the bootstrap record outliving an interrupted install (Slice 8).
 */
import { invoke } from "@tauri-apps/api/core";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "./App";
import { catalogue, job, model, overview } from "./test-fixtures";
import type { BootstrapState, JobStatus, Overview } from "./types";

const mockInvoke = vi.mocked(invoke);

type Backend = {
  overview: () => Overview | Promise<Overview>;
  jobStatus?: () => JobStatus;
  forget?: () => unknown;
};

function backend(handlers: Backend) {
  mockInvoke.mockImplementation(async (command: string) => {
    switch (command) {
      case "overview":
        return await handlers.overview();
      case "config_read":
        return {};
      case "models_status":
        return catalogue([
          model({
            key: "local-abc",
            display_name: "My local model",
            provenance: "imported_local",
            can_download: false,
          }),
        ]);
      case "job_status":
        return handlers.jobStatus ? handlers.jobStatus() : job();
      case "local_model_forget":
        return handlers.forget ? handlers.forget() : { ok: true };
      default:
        throw new Error(`unexpected command ${command}`);
    }
  });
}

beforeEach(() => mockInvoke.mockReset());
afterEach(() => vi.restoreAllMocks());

it("keeps an action's result when the background poll starts failing", async () => {
  // The bug this exists for: one shared `error`, cleared by a four-second poll,
  // erased every message an action produced. The two channels must now be able
  // to hold different things at the same time.
  let healthy = true;
  backend({
    overview: () => {
      if (!healthy) throw new Error("data directory not found");
      return overview();
    },
    forget: () => ({ ok: false, reason: "This model is the current default model." }),
  });

  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("tab", { name: /Models/ }));
  await user.click(await screen.findByRole("button", { name: /Forget/ }));
  const refusal = await screen.findByText("This model is the current default model.");

  // Now the poll breaks. It runs every four seconds; several cycles pass.
  healthy = false;
  await waitFor(() => expect(screen.getByText(/Background status check failed/)).toBeTruthy(), {
    timeout: 15000,
  });

  // Both are on screen, and distinguishable.
  expect(refusal.isConnected).toBe(true);
  expect(screen.getByText("This model is the current default model.")).toBeTruthy();
}, 20000);

it("still shows a running job after leaving Models and coming back", async () => {
  // The job belongs to Rust and outlives the panel. Before the state was lifted
  // out of `Models`, returning to the view showed nothing until the next poll.
  backend({
    overview: () => overview({ server: { running: true, port: 8765, lastExit: null } }),
    jobStatus: () =>
      job({ state: "running", kind: "prequantize", target: "flux2-dev", message: "converting" }),
  });

  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("tab", { name: /Models/ }));
  expect(await screen.findByText("Conversion")).toBeTruthy();

  await user.click(screen.getByRole("tab", { name: /Logs/ }));
  await user.click(screen.getByRole("tab", { name: /Models/ }));

  // Present on the very first render after remounting, with no tick in between.
  expect(screen.getByText("Conversion")).toBeTruthy();
  expect(screen.getByText("flux2-dev")).toBeTruthy();
});

it.each([
  ["uninitialized", "Install", "Install the environment"],
  ["updateRequired", "Rebuild", "Update the environment"],
  ["broken", "Repair", "Repair the environment"],
] as [BootstrapState, string, string][])(
  "shows Setup as %s with a %s action",
  async (state, action, title) => {
    // The distinction the old marker could not express: `broken` is an install
    // that was interrupted, and offering "Install" there would describe the
    // machine as fresh when it is not.
    backend({
      overview: () =>
        overview({
          bootstrap: {
            ready: false,
            state,
            installedVersion: state === "uninitialized" ? null : "1.0.0",
            appVersion: "1.0.0",
            envPath: "/data/env",
            failure: null,
          },
        }),
    });

    render(<App />);

    expect(await screen.findByText(title)).toBeTruthy();
    expect(screen.getByRole("button", { name: action })).toBeTruthy();
    // Exclusive: no view tabs while the runtime is not ready.
    expect(screen.queryByRole("tab", { name: /Models/ })).toBeNull();
  },
);

it("reports what the last interrupted install said", async () => {
  backend({
    overview: () =>
      overview({
        bootstrap: {
          ready: false,
          state: "broken",
          installedVersion: "1.0.0",
          appVersion: "1.0.0",
          envPath: "/data/env",
          failure: "uv sync failed (code Some(1))",
        },
      }),
  });

  render(<App />);
  expect(await screen.findByText(/uv sync failed/)).toBeTruthy();
});

it("does not offer per-model controls in Configuration as well as Models", async () => {
  // The duplication this removes: the same model appeared twice under two names,
  // its availability and conversion in one view and its `enabled`/quantize/steps
  // in a table under the other. Ownership is now one place, and the test asserts
  // the *absence* in the view that gave it up.
  backend({ overview: () => overview() });

  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("tab", { name: /Configuration/ }));
  await screen.findByText(/Generation defaults/i);

  expect(screen.queryByRole("switch")).toBeNull();
  expect(screen.queryByLabelText(/quantization for /i)).toBeNull();
  expect(screen.queryByLabelText(/^Steps for /)).toBeNull();
  expect(screen.queryByLabelText(/^Guidance for /)).toBeNull();
  expect(screen.queryByLabelText(/edits endpoint for /i)).toBeNull();

  // The reverse split, made in this slice: one account-wide secret, and it is
  // configured here rather than beside a catalogue of per-model controls.
  expect(screen.getByLabelText("Hugging Face token")).toBeTruthy();
  // And it says where they went, rather than leaving a hole.
  expect(screen.getByText(/live on each model's row/i)).toBeTruthy();

  // Server-wide settings stay: this moved model settings, not everything.
  expect(screen.getByLabelText(/^Port$/)).toBeTruthy();
  expect(screen.getByLabelText(/Hugging Face model directory/i)).toBeTruthy();

  // The same controls exist on the model's row in Models.
  await user.click(screen.getByRole("tab", { name: /Models/ }));
  expect(await screen.findByRole("switch", { name: /Enable/ })).toBeTruthy();
  // And the token is not offered a second time here: one field, one owner.
  expect(screen.queryByLabelText("Hugging Face token")).toBeNull();
});
