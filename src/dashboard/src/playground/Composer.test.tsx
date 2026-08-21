/**
 * The composer's conditional controls, driven by the model's own declaration.
 *
 * The rule this pins is the one the whole dashboard follows: what a model can do
 * is the backend's to say, published per model at `/v1/models/{id}`, and the
 * page's job is to disable the control and explain why — never to keep a table
 * of which models support what. `test_react_keeps_no_quantization_table_of_its
 * _own` on the server side fails the build if that rule is broken.
 *
 * Deliberately fail-closed: until the answer arrives, the control is inert. An
 * option offered before it is known to exist is an option the server will
 * refuse, after the user has typed into it.
 */

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { type FakeServer, fakeServer } from "../test-server";
import type { ModelCapabilities } from "../types";
import { Composer, type Draft } from "./Composer";

let server: FakeServer;

const MODELS = [
  { id: "with-cfg", name: "with-cfg" },
  { id: "distilled", name: "distilled" },
];

function capabilities(patch: Partial<ModelCapabilities> = {}): ModelCapabilities {
  return {
    repo: "example/model",
    default_size: "512x512",
    default_steps: 6,
    default_guidance: 4.5,
    quantize: null,
    active_variant: null,
    supports_quantization: false,
    quantize_choices: [],
    supports_prequantize: false,
    prequantize_choices: [],
    prequantize_strategy: null,
    quantization_note: null,
    license: "Apache-2.0",
    gated: false,
    prompt_formats: ["text"],
    preset: null,
    min_dimension: 64,
    max_dimension: 2048,
    scheduler: "euler",
    supports_guidance: true,
    supports_negative_prompt: true,
    supports_image_to_image: false,
    supports_edit: false,
    ...patch,
  };
}

beforeEach(() => {
  server = fakeServer();
  server.on("GET /v1/models/with-cfg", () => ({
    id: "with-cfg",
    object: "model",
    created: 0,
    owned_by: "qds",
    mflux: capabilities(),
  }));
  server.on("GET /v1/models/distilled", () => ({
    id: "distilled",
    object: "model",
    created: 0,
    owned_by: "qds",
    mflux: capabilities({ supports_negative_prompt: false, supports_guidance: false }),
  }));
});

afterEach(() => server.restore());

function composer(onSubmit: (draft: Draft) => void) {
  return render(
    <Composer
      models={MODELS}
      defaultModel="with-cfg"
      maxN={4}
      busy={false}
      error={null}
      onSubmit={onSubmit}
    />,
  );
}

async function openAdvanced() {
  await userEvent.click(screen.getByRole("button", { name: "Advanced settings" }));
  return screen.getByRole("dialog", { name: "Advanced settings" });
}

it("offers the negative prompt for a model that declares one", async () => {
  composer(vi.fn());
  await waitFor(() => expect(server.requests.some((r) => r.path.includes("with-cfg"))).toBe(true));
  await openAdvanced();

  const field = screen.getByLabelText("Negative prompt") as HTMLTextAreaElement;
  await waitFor(() => expect(field.disabled).toBe(false));
  await act(async () => {});
});

it("greys it out for a model with no unconditional branch, and says why", async () => {
  composer(vi.fn());
  await userEvent.selectOptions(screen.getByLabelText("Model"), "distilled");
  await openAdvanced();

  const field = screen.getByLabelText("Negative prompt") as HTMLTextAreaElement;
  await waitFor(() => expect(field.disabled).toBe(true));
  // Disabled *and* explained: a dead control with no reason is a bug report.
  expect(screen.getByText(/guidance-distilled/)).toBeTruthy();
  await act(async () => {});
});

it("sends the negative prompt with the request", async () => {
  const onSubmit = vi.fn();
  composer(onSubmit);
  await openAdvanced();
  const field = screen.getByLabelText("Negative prompt");
  await waitFor(() => expect((field as HTMLTextAreaElement).disabled).toBe(false));

  await userEvent.type(field, "blurry, watermark");
  await userEvent.type(screen.getByLabelText("Prompt"), "a fox");
  await userEvent.click(screen.getByRole("button", { name: "Generate ↵" }));

  expect(onSubmit).toHaveBeenCalledWith(
    expect.objectContaining({ negativePrompt: "blurry, watermark" }),
  );
  await act(async () => {});
});

it("keeps what was typed but stops sending it when the model changes", async () => {
  // The text is not erased: it is expensive to retype and the disabled field
  // cannot send it by accident, so switching back must restore it.
  const onSubmit = vi.fn();
  composer(onSubmit);
  await openAdvanced();
  const field = screen.getByLabelText("Negative prompt");
  await waitFor(() => expect((field as HTMLTextAreaElement).disabled).toBe(false));
  await userEvent.type(field, "blurry");

  // Reaching the model picker closes the popover, as a press anywhere outside
  // it does; reopening is what the user does too.
  await userEvent.selectOptions(screen.getByLabelText("Model"), "distilled");
  await openAdvanced();
  await waitFor(() =>
    expect((screen.getByLabelText("Negative prompt") as HTMLTextAreaElement).disabled).toBe(true),
  );
  expect((screen.getByLabelText("Negative prompt") as HTMLTextAreaElement).value).toBe("blurry");

  await userEvent.type(screen.getByLabelText("Prompt"), "a fox");
  await userEvent.click(screen.getByRole("button", { name: "Generate ↵" }));

  expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ negativePrompt: null }));
  await act(async () => {});
});

it("stays inert while the model's capabilities are unknown", async () => {
  server.fail("GET /v1/models/with-cfg", 503, "Model catalogue unavailable.");
  composer(vi.fn());
  await openAdvanced();

  const field = screen.getByLabelText("Negative prompt") as HTMLTextAreaElement;
  expect(field.disabled).toBe(true);
  await act(async () => {});
});
