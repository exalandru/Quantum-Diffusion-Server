/**
 * The shell: what survives navigation, and what survives a failing poll.
 *
 * Both are about state *outliving* a render — an action result outliving a
 * background poll (Slice 1), and a server-owned job outliving a view switch
 * (Slice 2). The shell reaches its server over `fetch` and nothing else, so
 * that is the only seam these replace (`test-server.ts`).
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "./App";
import { catalogue, job, model, overview } from "./test-fixtures";
import { type FakeServer, fakeServer } from "./test-server";
import type { Capabilities, Health, JobStatus, Overview } from "./types";

let server: FakeServer;

const HEALTH: Health = {
  status: "ok",
  version: "1.0.0",
  default_model: "z-image-turbo",
  models: ["z-image-turbo"],
  loaded_model: null,
  idle_unload_s: null,
  memory: {},
};

// `models` is empty on purpose: the rows here come from `/admin/models`, and a
// row whose live capabilities are unknown is a real state the panel must render
// rather than crash on.
const CAPABILITIES: Capabilities = {
  default_model: "z-image-turbo",
  max_n: 4,
  response_formats: ["b64_json"],
  models: {},
  rewrite: {
    available: false,
    reason: "no rewriter configured",
    downloaded: false,
    sizeMb: null,
    word_ceiling: 60,
  },
};

type Backend = {
  overview: () => Overview;
  jobStatus?: () => JobStatus;
  forget?: () => unknown;
};

/**
 * Every route the shell touches while it is up.
 *
 * Spelled out rather than defaulted: an unrouted request is a failure in
 * `test-server.ts`, which is what makes a panel quietly calling something
 * nobody declared visible here instead of silently absorbed.
 */
function backend(handlers: Backend) {
  server.on("GET /admin/overview", () => handlers.overview());
  server.on("GET /health", () => HEALTH);
  server.on("GET /admin/config", () => ({}));
  server.on("GET /admin/logs", () => ({ entries: [], lastSeq: 0, dropped: 0 }));
  server.on("GET /admin/jobs", () => (handlers.jobStatus ? handlers.jobStatus() : job()));
  server.on("GET /admin/models", () =>
    catalogue([
      model({
        key: "local-abc",
        display_name: "My local model",
        provenance: "imported_local",
        can_download: false,
      }),
    ]),
  );
  server.on("POST /admin/import/forget", () =>
    handlers.forget ? handlers.forget() : { ok: true },
  );
  server.on("GET /v1/capabilities", () => CAPABILITIES);
  // A body that never produces a frame: the shell's panels subscribe, and this
  // test is not about what the stream says.
  server.on("GET /v1/progress", () => new Response(new ReadableStream({ start() {} })));
}

beforeEach(() => {
  server = fakeServer();
});
afterEach(() => {
  server.restore();
  vi.restoreAllMocks();
});

it("keeps an action's result when the background poll starts failing", async () => {
  // The bug this exists for: one shared `error`, cleared by a four-second poll,
  // erased every message an action produced. The two channels must now be able
  // to hold different things at the same time.
  let healthy = true;
  backend({
    overview: () => overview(),
    forget: () => ({ ok: false, reason: "This model is the current default model." }),
  });
  server.on("GET /admin/overview", () => {
    if (!healthy) {
      return new Response(
        JSON.stringify({ error: { message: "data directory not found", type: "server_error" } }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      );
    }
    return overview();
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
  // The job belongs to the server and outlives the panel. Before the state was
  // lifted out of `Models`, returning to the view showed nothing until the next
  // poll.
  backend({
    overview: () => overview(),
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

  // Not "no switches at all": Configuration owns one of its own now, the
  // local-network toggle. What must not be here is a *model's* switch.
  expect(screen.queryByRole("switch", { name: /Enable/ })).toBeNull();
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
