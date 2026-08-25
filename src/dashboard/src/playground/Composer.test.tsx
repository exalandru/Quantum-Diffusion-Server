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
import type { ModelCapabilities, RewriteCapabilities } from "../types";
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

function composer(
  onSubmit: (draft: Draft) => void,
  extra: { rewrite?: RewriteCapabilities | null; presetPrompt?: { text: string; nonce: number } } = {},
) {
  return render(
    <Composer
      models={MODELS}
      defaultModel="with-cfg"
      maxN={4}
      busy={false}
      error={null}
      rewrite={extra.rewrite ?? null}
      presetPrompt={extra.presetPrompt ?? null}
      onSubmit={onSubmit}
    />,
  );
}

const REWRITE_ON: RewriteCapabilities = {
  available: true,
  reason: null,
  downloaded: true,
  sizeMb: 2263,
  word_ceiling: 40,
};

async function openAdvanced() {
  await userEvent.click(screen.getByRole("button", { name: "Advanced settings" }));
  return screen.getByRole("dialog", { name: "Advanced settings" });
}

/** Shut the settings dialog, the way a user does before typing a prompt.
 *
 * The panel is a modal now, not a popover: RAC contains focus in it and marks
 * the rest of the page hidden while it is up, so the composer's own fields are
 * deliberately unreachable until it is closed. That is the mechanism working,
 * and the tests below say so explicitly rather than reaching through it. */
async function closeAdvanced() {
  await userEvent.click(screen.getByRole("button", { name: "Close" }));
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
  await closeAdvanced();
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
  await closeAdvanced();

  // The model picker with the dialog shut, then the dialog again: what the user
  // does, and the point is that the text survived the round trip.
  await userEvent.selectOptions(screen.getByLabelText("Model"), "distilled");
  await openAdvanced();
  await waitFor(() =>
    expect((screen.getByLabelText("Negative prompt") as HTMLTextAreaElement).disabled).toBe(true),
  );
  expect((screen.getByLabelText("Negative prompt") as HTMLTextAreaElement).value).toBe("blurry");

  await closeAdvanced();
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


// ── Enhance ────────────────────────────────────────────────────────────────
//
// Same rule as every other conditional control here: the server says what it
// offers, and the page's job is to hide or explain -- never to keep its own
// idea of when rewriting is possible.

it("offers no Enhance control when the server does not rewrite", async () => {
  composer(vi.fn());
  await waitFor(() => screen.getByRole("button", { name: "Advanced settings" }));
  expect(screen.queryByRole("button", { name: /enhance/i })).toBeNull();
});

it("offers Enhance when the server does rewrite", async () => {
  composer(vi.fn(), { rewrite: REWRITE_ON });
  await waitFor(() => expect(screen.getByRole("button", { name: /enhance/i })).toBeTruthy());
});

it("hides Enhance for a model that takes only a JSON caption", async () => {
  server.on("GET /v1/models/with-cfg", () => ({
    id: "with-cfg",
    object: "model",
    created: 0,
    owned_by: "qds",
    mflux: capabilities({ prompt_formats: ["json"] }),
  }));
  composer(vi.fn(), { rewrite: REWRITE_ON });
  await waitFor(() => screen.getByRole("button", { name: "Advanced settings" }));
  // The server refuses rewriting outright for these, so the control would be a
  // button whose only outcome is an error.
  expect(screen.queryByRole("button", { name: /enhance/i })).toBeNull();
});

it("asks for a rewrite only when Enhance is on", async () => {
  const onSubmit = vi.fn();
  composer(onSubmit, { rewrite: REWRITE_ON });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));

  await userEvent.type(screen.getByLabelText("Prompt"), "un chat sur un toit");
  await userEvent.click(screen.getByRole("button", { name: /generate/i }));
  expect(onSubmit.mock.calls[0]![0].rewrite).toBe(false);

  await userEvent.type(screen.getByLabelText("Prompt"), "un chat sur un toit");
  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));
  await userEvent.click(screen.getByRole("button", { name: /generate/i }));
  expect(onSubmit.mock.calls[1]![0].rewrite).toBe(true);
});

it("warns before submitting that a long prompt is generated as typed", async () => {
  composer(vi.fn(), { rewrite: REWRITE_ON });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));
  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));

  await userEvent.type(screen.getByLabelText("Prompt"), "short prompt");
  expect(screen.queryByText(/already detailed/i)).toBeNull();

  // The ceiling is the server's, published so this line cannot drift from what
  // the route enforces -- and said *before* submitting, not after.
  await act(async () => {
    await userEvent.clear(screen.getByLabelText("Prompt"));
  });
  await userEvent.type(screen.getByLabelText("Prompt"), "word ".repeat(45));
  await waitFor(() => expect(screen.getByText(/already detailed/i)).toBeTruthy());
});

it("takes an enhanced prompt into the box with Enhance switched off", async () => {
  const onSubmit = vi.fn();
  const view = composer(onSubmit, { rewrite: REWRITE_ON });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));
  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));

  // "Use this prompt" in the feed. Enhancing an already-enhanced prompt is the
  // one case measured to make things worse, so accepting one turns it off.
  view.rerender(
    <Composer
      models={MODELS}
      defaultModel="with-cfg"
      maxN={4}
      busy={false}
      error={null}
      rewrite={REWRITE_ON}
      presetPrompt={{ text: "a ginger cat on terracotta tiles at dusk", nonce: 1 }}
      onSubmit={onSubmit}
    />,
  );
  await waitFor(() =>
    expect((screen.getByLabelText("Prompt") as HTMLTextAreaElement).value).toBe(
      "a ginger cat on terracotta tiles at dusk",
    ),
  );
  await userEvent.click(screen.getByRole("button", { name: /generate/i }));
  expect(onSubmit.mock.calls[0]![0].rewrite).toBe(false);
});


it("warns that a first Enhance will download the rewriter", async () => {
  composer(vi.fn(), { rewrite: { ...REWRITE_ON, downloaded: false } });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));

  // Nothing before it is switched on: an install note for a control nobody
  // asked for is noise.
  expect(screen.queryByText(/first use downloads/i)).toBeNull();

  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));
  // Same warning `UpscalePopover` gives, for the same reason: a first Enhance
  // fetches a gigabyte and a silent control looks like it has hung.
  await waitFor(() => expect(screen.getByText(/first use downloads 2263 MB/i)).toBeTruthy());
});

it("says nothing about downloading once the rewriter is present", async () => {
  composer(vi.fn(), { rewrite: REWRITE_ON });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));
  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));
  expect(screen.queryByText(/first use downloads/i)).toBeNull();
});


it("never names the model doing the rewriting", async () => {
  // Which LLM improves the prompt is an operator fact, not a user fact. The
  // field is absent from the type, so this catches a re-introduction through
  // some other string rather than through the capability payload.
  const { container } = composer(vi.fn(), { rewrite: { ...REWRITE_ON, downloaded: false } });
  await waitFor(() => screen.getByRole("button", { name: /enhance/i }));
  await userEvent.click(screen.getByRole("button", { name: /enhance/i }));

  const shown = container.textContent ?? "";
  const titles = Array.from(container.querySelectorAll("[title]"))
    .map((el) => el.getAttribute("title") ?? "")
    .join(" ");
  for (const name of ["Qwen", "qwen", "Mistral", "Ministral", "Llama", "1.7B", "4B"]) {
    expect(shown).not.toContain(name);
    expect(titles).not.toContain(name);
  }
});
