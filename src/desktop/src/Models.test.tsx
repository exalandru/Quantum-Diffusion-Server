/**
 * What the Models view may and may not offer.
 *
 * These are the conditionals the redesign could most easily get wrong, because
 * each one is a rule the *backend* owns and the interface is only allowed to
 * render. A test that passed by re-deriving the rule in React would defeat its
 * own purpose, so each case pins a field the backend publishes and asserts the
 * interface followed it — including the cases where following it means offering
 * nothing at all.
 */
import { invoke } from "@tauri-apps/api/core";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Models } from "./panels/Models";
import type { JobView } from "./job";
import { catalogue, components, disk, job, model, overview } from "./test-fixtures";
import type { JobStatus, ModelStatus } from "./types";

const mockInvoke = vi.mocked(invoke);
const onConfigChanged = vi.fn(async () => {});

function idleJobs(patch: Partial<JobView> = {}): JobView {
  return {
    job: job(),
    error: null,
    active: false,
    refresh: vi.fn(async () => {}),
    onSettled: () => () => {},
    ...patch,
  };
}

function show(
  models: ModelStatus[],
  jobs: JobView = idleJobs(),
  config: unknown = {},
  state = overview(),
) {
  mockInvoke.mockImplementation(async (command: string) => {
    if (command === "models_status") return catalogue(models);
    if (command === "local_model_forget") return { ok: true };
    throw new Error(`unexpected command ${command}`);
  });
  return render(
    <Models
      state={state}
      client={null}
      config={config}
      jobs={jobs}
      onConfigChanged={onConfigChanged}
    />,
  );
}

/** The row for one model, so a query cannot accidentally match its neighbour. */
async function row(name: string) {
  const heading = await screen.findByText(name);
  const element = heading.closest("li");
  if (!element) throw new Error(`no row for ${name}`);
  return within(element);
}

/** Open one of the row's dialogs and return a scope over it. */
async function openDialog(name: string, action: "Quantization…" | "Model settings…") {
  const it_ = await row(name);
  await userEvent.click(it_.getByRole("button", { name: action }));
  return within(it_.getByRole("dialog"));
}

/** The model names a list carries, in the order it carries them. */
function namesIn(list: HTMLElement): string[] {
  return within(list)
    .getAllByRole("heading")
    .map((heading) => heading.textContent ?? "");
}

beforeEach(() => mockInvoke.mockReset());
afterEach(() => vi.restoreAllMocks());

describe("what may be offered", () => {
  it("never offers a HuggingFace download for an imported local model", async () => {
    // The discriminating part: this model is *missing*, which for a built-in is
    // exactly the state that offers Install. Provenance is what forbids it, and
    // the backend says so through `can_download` rather than through the shape of
    // the path.
    show([
      model({
        key: "local-abc",
        display_name: "My local model",
        provenance: "imported_local",
        availability: "missing",
        can_download: false,
        repo: "/Volumes/Models/z-image",
        base_profile_key: "z-image-turbo",
      }),
    ]);

    const it_ = await row("My local model");
    expect(it_.queryByRole("button", { name: /install|resume/i })).toBeNull();
    expect(it_.getByRole("button", { name: /forget/i })).toBeTruthy();
  });

  it("does not offer to re-download weights whose volume is merely unplugged", async () => {
    // The expensive mistake this prevents: the weights are not gone, the disk is.
    // `can_download` is true here, so only the availability can be what stops it.
    show([
      model({
        availability: "volume_unmounted",
        can_download: true,
        detail: "/Volumes/Big is not mounted.",
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByRole("button", { name: /install|resume/i })).toBeNull();
    expect(it_.getByText(/volume unavailable/i)).toBeTruthy();
    // And it says what to do about it, in the backend's words.
    expect(it_.getByText("/Volumes/Big is not mounted.")).toBeTruthy();
  });

  it("offers Resume, not Install, for an interrupted download", async () => {
    show([model({ availability: "partial", can_download: true })]);

    const it_ = await row("z-image-turbo");
    expect(it_.getByRole("button", { name: "Resume" })).toBeTruthy();
    expect(it_.queryByRole("button", { name: "Install" })).toBeNull();
  });

  it("offers no conversion for a model the backend says cannot be pre-quantized", async () => {
    show([model({ quantization: { ...model().quantization, supports_prequantize: false } })]);

    const it_ = await row("z-image-turbo");
    // Not in the row under any circumstances now, but the discriminating part is
    // inside the dialog: this model *can* be quantized on load, so the dialog
    // exists and must still refuse to offer a conversion.
    expect(it_.queryByRole("button", { name: /pre-quantize/i })).toBeNull();
    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.getByLabelText("Runtime quantization for z-image-turbo")).toBeTruthy();
    expect(dialog.queryByRole("button", { name: /pre-quantize/i })).toBeNull();
    expect(dialog.queryByLabelText(/bit depth/i)).toBeNull();
  });
});

describe("the row's hierarchy", () => {
  it("gives the name an identity of its own, above the state badges", async () => {
    // What the redesign is for: the name used to sit on one baseline with every
    // pill the row could produce, so the thing you scan for was the shortest
    // item in a line of seven.
    show([
      model({
        display_name: "Z-Image Turbo",
        availability: "present",
        size_gb: 12.5,
        disk: disk({ source_bytes: 12_500_000_000, active_bytes: 12_500_000_000, total_bytes: 12_500_000_000 }),
      }),
    ]);

    const it_ = await row("Z-Image Turbo");
    const heading = it_.getByRole("heading", { name: "Z-Image Turbo" });
    const header = heading.parentElement as HTMLElement;

    // The header holds the identity and the one control that acts on the whole
    // model. No state pill shares its baseline.
    expect(within(header).getByRole("switch", { name: /^Enable/ })).toBeTruthy();
    expect(header.querySelector(".pill")).toBeNull();

    // State lives in its own band below, where availability and size are two
    // separate facts rather than one badge standing in for the other.
    const badges = (heading.closest("li") as HTMLElement).querySelector(
      ".model-badges",
    ) as HTMLElement;
    expect(within(badges).getByText("installed")).toBeTruthy();
    expect(within(badges).getByText("12.5 GB")).toBeTruthy();
  });

  it("keeps the technical controls out of the row", async () => {
    show([
      model({
        quantization: {
          ...model().quantization,
          supports_prequantize: true,
          prequantize_components: components("transformer", "text_encoder", "vae"),
          prequantize_choices: [4],
          prequantize_strategy: "mflux_save",
        },
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByRole("combobox")).toBeNull();
    expect(it_.queryByRole("button", { name: /pre-quantize/i })).toBeNull();
    expect(it_.getByRole("button", { name: "Quantization…" })).toBeTruthy();
    expect(it_.getByRole("button", { name: "Model settings…" })).toBeTruthy();
  });
});

describe("a configuration the server would refuse", () => {
  const warning = {
    code: "default_model_disabled",
    field: "default_model",
    message: 'Default model "z-image-turbo" is disabled. Enable it or choose another default model.',
  };

  function showWith(models: ModelStatus[], warnings = [warning]) {
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue(models, warnings);
      throw new Error(`unexpected command ${command}`);
    });
    return render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );
  }

  it("renders every row and reports the problem beside them", async () => {
    // The defect: this state made `--status` fail, so the catalogue became a
    // traceback — taking away the switches that repair it.
    showWith([model({ key: "z-image-turbo" }), model({ key: "z-image", display_name: "Z-Image" })]);

    expect(await screen.findByText("z-image-turbo")).toBeTruthy();
    expect(screen.getByText("Z-Image")).toBeTruthy();
    expect(screen.getByText(/Default model "z-image-turbo" is disabled/)).toBeTruthy();
    // A warning, not the catalogue's failure state.
    expect(screen.queryByText(/The catalogue could not be read/)).toBeNull();
    expect(screen.queryByText(/Traceback/)).toBeNull();
  });

  it("says nothing when the configuration is sound", async () => {
    showWith([model()], []);
    await screen.findByText("z-image-turbo");
    expect(screen.queryByText(/Default model/)).toBeNull();
  });

  it("still shows a catalogue that genuinely could not be read", async () => {
    // The other channel is untouched: an unreadable file is not a warning.
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") throw new Error("server-config.json is not valid JSON");
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );
    await waitFor(() =>
      expect(screen.getByText(/server-config.json is not valid JSON/)).toBeTruthy(),
    );
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });

  it("surfaces the backend's refusal when a switch would break the invariant", async () => {
    // The write is refused at the boundary that owns the file; the row shows the
    // reason and the model stays enabled, because nothing was written.
    const refusal =
      '"z-image-turbo" is currently the default model. Choose another default model in ' +
      "Configuration before disabling it.";
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue([model()], []);
      if (command === "config_write") throw new Error(refusal);
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{ default_model: "z-image-turbo", models: {} }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("switch", { name: "Enable z-image-turbo" }));

    await waitFor(() => expect(screen.getByText(refusal)).toBeTruthy());
    // Verbatim, and the switch is still on: nothing was written.
    expect(it_.getByRole("switch").getAttribute("aria-checked")).toBe("true");
  });
});

describe("when a conversion finishes", () => {
  /** A jobs view whose settle edge the test fires by hand. */
  function settlable(job_: JobStatus) {
    const listeners = new Set<(settled: JobStatus) => void>();
    const view: JobView = {
      job: job_,
      error: null,
      active: false,
      refresh: vi.fn(async () => {}),
      onSettled: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
    return { view, settle: (settled: JobStatus) => listeners.forEach((l) => l(settled)) };
  }

  const converted = (patch: Partial<ModelStatus> = {}) =>
    model({
      quantization: {
        ...model().quantization,
        supports_prequantize: true,
        prequantize_choices: [4],
        prequantize_strategy: "qds_memory_bounded",
        prequantize_components: components("transformer", "vae"),
      },
      ...patch,
    });

  const done = (bits: number) =>
    job({
      state: "completed",
      kind: "prequantize",
      target: `z-image-turbo @ ${bits}-bit`,
      event: "prequantize_done",
      fields: { model: "z-image-turbo", variant_ready: true, bits, components: ["transformer", "vae"] },
      finishedAtMs: 1000,
    });

  const partial = (bits: number) =>
    job({
      state: "completed",
      kind: "prequantize",
      target: `z-image-turbo @ ${bits}-bit`,
      event: "prequantize_partial",
      fields: {
        model: "z-image-turbo",
        variant_ready: false,
        bits,
        completed: ["transformer"],
        missing: ["vae"],
      },
      finishedAtMs: 1000,
    });

  /** Renders with a catalogue that changes when the backend is asked again. */
  function showChanging(before: ModelStatus[], after: ModelStatus[], job_: JobStatus) {
    const reads: string[] = [];
    let current = before;
    mockInvoke.mockImplementation(async (command: string) => {
      reads.push(command);
      if (command === "models_status") return catalogue(current);
      throw new Error(`unexpected command ${command}`);
    });
    const { view, settle } = settlable(job_);
    const onChanged = vi.fn(async () => {});
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={view}
        onConfigChanged={onChanged}
      />,
    );
    return {
      reads,
      onChanged,
      settle: async (settled: JobStatus) => {
        current = after;
        await act(async () => settle(settled));
      },
    };
  }

  it("re-reads the catalogue and the configuration, rather than trusting the job", async () => {
    // The defect this slice exists for: the settle handler only refreshed for
    // `kind === "fetch"`, so a finished conversion left every fact about it
    // unasked-for until the panel was remounted.
    const { reads, onChanged, settle } = showChanging([converted()], [converted()], done(4));
    await screen.findByText("z-image-turbo");
    const before = reads.filter((command) => command === "models_status").length;

    await settle(done(4));

    expect(reads.filter((command) => command === "models_status").length).toBe(before + 1);
    // And the configuration, which is where the supervisor recorded the
    // selection — the catalogue alone would leave `App` holding the old one.
    expect(onChanged).toHaveBeenCalled();
  });

  it("shows the new variant without the panel being remounted", async () => {
    const { settle } = showChanging(
      [converted({ partials: [{ bits: 4, path: "/a", strategy: null, components: { transformer: "complete", vae: "missing" }, size_bytes: 1 }] })],
      [
        converted({
          variants: [{ bits: 4, path: "/a", strategy: null, legacy: false, size_bytes: 5_900_000_000 }],
          active_variant: 4,
        }),
      ],
      done(4),
    );

    const it_ = await row("z-image-turbo");
    expect(it_.queryByText("using the 4-bit copy")).toBeNull();

    await settle(done(4));

    // Same row, no remount: the summary the backend now reports.
    expect((await row("z-image-turbo")).getByText("using the 4-bit copy")).toBeTruthy();
  });

  it("updates an open Quantization dialog in place", async () => {
    const { settle } = showChanging(
      [converted()],
      [
        converted({
          partials: [
            {
              bits: 4,
              path: "/a",
              strategy: null,
              components: { transformer: "complete", vae: "missing" },
              size_bytes: 1,
            },
          ],
        }),
      ],
      partial(4),
    );

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.queryByText("Converted")).toBeNull();

    await settle(partial(4));

    // The dialog was never closed, and it is showing the refreshed truth.
    const stillOpen = within((await row("z-image-turbo")).getByRole("dialog"));
    expect(stillOpen.getByText("Converted")).toBeTruthy();
    expect(stillOpen.getByText(/cannot be used until/i)).toBeTruthy();
    // Feedback where the operation was started from, not only in Logs.
    expect(stillOpen.getByText(/Transformer converted — 1 component remaining/)).toBeTruthy();
  });

  it("reports a finished variant as ready and selected", async () => {
    const { settle } = showChanging(
      [converted()],
      [
        converted({
          variants: [{ bits: 4, path: "/a", strategy: null, legacy: false, size_bytes: 5_900_000_000 }],
          active_variant: 4,
        }),
      ],
      done(4),
    );
    await screen.findByText("z-image-turbo");

    await settle(done(4));

    expect(screen.getByText("4-bit variant ready and selected.")).toBeTruthy();
  });

  it("says a partial run is not a variant, and never claims one is selected", async () => {
    const { settle } = showChanging([converted()], [converted()], partial(4));
    await screen.findByText("z-image-turbo");

    await settle(partial(4));

    expect(screen.getByText(/Transformer converted — 1 component remaining/)).toBeTruthy();
    expect(screen.queryByText(/ready and selected/)).toBeNull();
  });

  it("tells the truth about a running server that has not adopted the variant", async () => {
    // The configuration selects 4-bit; the running process loaded the source.
    // Nothing restarts it, and nothing pretends it is already using the copy.
    const capabilities = {
      default_model: "z-image-turbo",
      max_n: 4,
      response_formats: ["url"],
      models: { "z-image-turbo": { active_variant: null } },
    };
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status")
        return catalogue([
          converted({
            active_variant: 4,
            variants: [
              { bits: 4, path: "/a", strategy: null, legacy: false, size_bytes: 5_900_000_000 },
            ],
          }),
        ]);
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview({ server: { running: true, port: 8765, lastExit: null } })}
        client={{ capabilities: async () => capabilities } as never}
        config={{ models: { "z-image-turbo": { enabled: true } } }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await waitFor(() => expect(it_.getByText("restart required")).toBeTruthy());
    // No server command was issued by any of this.
    expect(
      (mockInvoke.mock.calls as unknown[][]).some(
        (call) => typeof call[0] === "string" && String(call[0]).startsWith("server_"),
      ),
    ).toBe(false);
  });

  it("says nothing about restarting once the running server agrees", async () => {
    const capabilities = {
      default_model: "z-image-turbo",
      max_n: 4,
      response_formats: ["url"],
      models: { "z-image-turbo": { active_variant: 4 } },
    };
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue([converted({ active_variant: 4 })]);
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview({ server: { running: true, port: 8765, lastExit: null } })}
        client={{ capabilities: async () => capabilities } as never}
        config={{ models: { "z-image-turbo": { enabled: true } } }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await waitFor(() => expect(it_.getByRole("switch")).toBeTruthy());
    expect(it_.queryByText("restart required")).toBeNull();
  });

  it("leaves every other saved variant alone", async () => {
    const existing = [
      { bits: 3, path: "/a/3", strategy: null, legacy: false, size_bytes: 3_900_000_000 },
      { bits: 4, path: "/a/4", strategy: null, legacy: false, size_bytes: 5_900_000_000 },
    ];
    const { settle } = showChanging(
      [converted({ variants: existing, active_variant: 4 })],
      [
        converted({
          variants: [
            ...existing,
            { bits: 8, path: "/a/8", strategy: null, legacy: false, size_bytes: 9_000_000_000 },
          ],
          active_variant: 8,
        }),
      ],
      done(8),
    );
    await screen.findByText("z-image-turbo");

    await settle(done(8));

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.getByRole("button", { name: /Use 3-bit/ })).toBeTruthy();
    expect(dialog.getByRole("button", { name: /Use 4-bit/ })).toBeTruthy();
    expect(dialog.getByRole("button", { name: /Use 8-bit · active/ })).toBeTruthy();
  });

  it("keeps a failed conversion's reason on screen, and refreshes nothing", async () => {
    const { reads, settle } = showChanging(
      [converted()],
      [converted()],
      job({
        state: "failed",
        kind: "prequantize",
        target: "z-image-turbo @ 4-bit",
        message: "the disk is full",
        finishedAtMs: 1000,
      }),
    );
    await screen.findByText("z-image-turbo");
    const before = reads.filter((command) => command === "models_status").length;

    await settle(
      job({ state: "failed", kind: "prequantize", target: "z-image-turbo @ 4-bit", message: "the disk is full", finishedAtMs: 1000 }),
    );

    expect(screen.getByText(/the disk is full/)).toBeTruthy();
    // Nothing changed on disk, so nothing was re-read.
    expect(reads.filter((command) => command === "models_status").length).toBe(before);
  });
});

describe("catalogue grouping", () => {
  it("separates gated from open built-ins, and keeps imported models apart from both", async () => {
    // Provenance and `gated` are backend facts; the grouping reads them and
    // reorders nothing inside a group.
    show([
      model({ key: "flux2-klein", display_name: "FLUX.2-klein", gated: true }),
      model({ key: "z-image", display_name: "Z-Image" }),
      model({ key: "fibo", display_name: "FIBO", gated: true }),
      model({ key: "z-image-turbo", display_name: "Z-Image Turbo" }),
      model({
        key: "local-abc",
        display_name: "My local model",
        provenance: "imported_local",
        can_download: false,
      }),
    ]);

    const gatedList = await screen.findByRole("list", { name: "Gated models" });
    expect(namesIn(gatedList)).toEqual(["FLUX.2-klein", "FIBO"]);
    expect(namesIn(screen.getByRole("list", { name: "Open models" }))).toEqual([
      "Z-Image",
      "Z-Image Turbo",
    ]);
    // The imported model is in neither catalogue group, whatever its state.
    expect(namesIn(screen.getByRole("list", { name: "Imported local models" }))).toEqual([
      "My local model",
    ]);
  });

  it("puts open models before gated ones, and local models after both", async () => {
    // What a reader meets first should be what they can install right now.
    show([
      model({ key: "flux2-klein", display_name: "FLUX.2-klein", gated: true }),
      model({ key: "z-image", display_name: "Z-Image" }),
      model({
        key: "local-abc",
        display_name: "My local model",
        provenance: "imported_local",
        can_download: false,
      }),
    ]);

    await screen.findByRole("list", { name: "Open models" });
    const lists = screen.getAllByRole("list").map((list) => list.getAttribute("aria-label"));
    expect(lists).toEqual(["Open models", "Gated models", "Imported local models"]);

    // Document order, not just presence: the headings read in the same order.
    const open = screen.getByRole("list", { name: "Open models" });
    const gated = screen.getByRole("list", { name: "Gated models" });
    expect(open.compareDocumentPosition(gated) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("shows no heading for a group the catalogue has nothing in", async () => {
    show([model()]);
    await screen.findByRole("list", { name: "Open models" });
    expect(screen.queryByRole("list", { name: "Gated models" })).toBeNull();
    expect(screen.queryByText("Gated models")).toBeNull();
  });
});

describe("quantization", () => {
  /** A model the backend says can do both kinds of quantization. */
  const both = (patch: Partial<ModelStatus> = {}) =>
    model({
      quantization: {
        supports_quantization: true,
        quantize_choices: [4, 8],
        supports_prequantize: true,
        prequantize_choices: [3, 4, 6],
        prequantize_strategy: "qds_memory_bounded",
        prequantize_components: components("transformer", "text_encoder", "vae"),
        note: null,
      },
      ...patch,
    });

  it("offers exactly the bit depths the backend published", async () => {
    // Not a list of its own: 5 and 8 are absent from `prequantize_choices` here
    // and must stay absent from the control driven by it.
    show([both()]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const select = dialog.getByLabelText(
      "Pre-quantized bit depth for z-image-turbo",
    ) as HTMLSelectElement;
    expect([...select.options].map((option) => option.value)).toEqual(["3", "4", "6"]);
  });

  it("keeps the runtime setting and the saved copy as two separate things", async () => {
    // The distinction the dialog exists to make: one is applied while the model
    // loads and writes nothing, the other writes a second copy of the weights.
    // They are published as two different lists and must be offered as two.
    show([both()]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const runtime = dialog.getByLabelText(
      "Runtime quantization for z-image-turbo",
    ) as HTMLSelectElement;
    const saved = dialog.getByLabelText(
      "Pre-quantized bit depth for z-image-turbo",
    ) as HTMLSelectElement;

    expect([...runtime.options].map((option) => option.value)).toEqual(["", "0", "4", "8"]);
    expect([...saved.options].map((option) => option.value)).toEqual(["3", "4", "6"]);
    // Two sections, named, and neither control is in the other's.
    expect(dialog.getByText("Runtime quantization")).toBeTruthy();
    expect(dialog.getByText("Pre-quantized copy")).toBeTruthy();
    expect(runtime.closest("fieldset")).not.toBe(saved.closest("fieldset"));
    expect(runtime.closest("fieldset")).not.toBeNull();
  });

  it("writes the runtime setting as its own key, touching no saved variant", async () => {
    const writes: unknown[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "models_status") return catalogue([both({ variants: [], active_variant: null })]);
      if (command === "config_write") {
        writes.push((args as { value?: unknown } | undefined)?.value);
        return null;
      }
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{ models: {} }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    await userEvent.selectOptions(
      dialog.getByLabelText("Runtime quantization for z-image-turbo"),
      "8",
    );

    await waitFor(() => expect(writes).toHaveLength(1));
    const written = (writes[0] as Record<string, any>).models["z-image-turbo"];
    expect(written.quantize).toBe(8);
    // Choosing how the model is quantized on load is not choosing a saved copy.
    expect(written.prequantized_variant).toBeUndefined();
  });

  it("marks which saved variant is active, and offers a way back to the source", async () => {
    // Source, saved variants and the active representation are three different
    // facts. The 4-bit variant exists *and* is in use; the 3-bit only exists.
    show([
      both({
        variants: [
          { bits: 3, path: "/a/3bit", strategy: "mflux_save", legacy: false, size_bytes: 3_900_000_000 },
          { bits: 4, path: "/a/4bit", strategy: "mflux_save", legacy: false, size_bytes: 5_900_000_000 },
        ],
        active_variant: 4,
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const active = dialog.getByRole("button", { name: /Use 4-bit · active/ });
    expect(active.getAttribute("aria-pressed")).toBe("true");
    expect((active as HTMLButtonElement).disabled).toBe(true);

    const inactive = dialog.getByRole("button", { name: "Use 3-bit" });
    expect(inactive.getAttribute("aria-pressed")).toBe("false");
    expect((inactive as HTMLButtonElement).disabled).toBe(false);

    expect(dialog.getByRole("button", { name: "Use original" })).toBeTruthy();
  });

  it("summarises the active variant in the row, without the controls", async () => {
    // The one fact from inside the dialog that changes what the model does.
    show([
      both({
        variants: [{ bits: 4, path: "/a/4bit", strategy: "mflux_save", legacy: false, size_bytes: 5_900_000_000 }],
        active_variant: 4,
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.getByText("using the 4-bit copy")).toBeTruthy();
    expect(it_.queryByRole("button", { name: /Use 4-bit/ })).toBeNull();
  });

  it("claims no variant for a model that has none, but still offers to make one", async () => {
    show([both()]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByText(/saved variants/i)).toBeNull();

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.queryByText(/saved variants/i)).toBeNull();
    expect(dialog.queryByRole("button", { name: "Use original" })).toBeNull();
    // The affordance is what says a variant *could* exist.
    expect(dialog.getByRole("button", { name: /pre-quantize selected/i })).toBeTruthy();
  });

  it("offers no quantization dialog at all where the backend supports neither kind", async () => {
    show([
      model({
        quantization: {
          supports_quantization: false,
          quantize_choices: [],
          supports_prequantize: false,
          prequantize_choices: [],
          prequantize_strategy: null,
          prequantize_components: [],
          note: "Loaded from a pre-quantized artifact.",
        },
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByRole("button", { name: "Quantization…" })).toBeNull();
    expect(it_.getByText("fixed precision")).toBeTruthy();
  });

  it("says a model's precision is fixed where the backend says quantizing is inert", async () => {
    show([
      model({
        quantization: {
          ...model().quantization,
          supports_quantization: false,
          quantize_choices: [],
          note: "Loaded from a pre-quantized artifact, whose stored precision mflux keeps.",
        },
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.getByText("fixed precision")).toBeTruthy();
  });
});

describe("feedback", () => {
  it("shows the backend's refusal when Forget would break the default model", async () => {
    const refusal =
      "This model is the current default model. Choose another default model in the " +
      "Configuration tab before removing this one from the library.";
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status")
        return catalogue([
          model({
            key: "local-abc",
            display_name: "My local model",
            provenance: "imported_local",
            can_download: false,
          }),
        ]);
      if (command === "local_model_forget") return { ok: false, code: "is_default_model", reason: refusal };
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("My local model");
    await userEvent.click(it_.getByRole("button", { name: /forget/i }));

    // Verbatim: reworded, it would stop naming the fix.
    await waitFor(() => expect(screen.getByText(refusal)).toBeTruthy());
    // And the row is still there, because nothing was removed.
    expect(screen.getByText("My local model")).toBeTruthy();
  });

  it("explains an unreadable catalogue instead of showing an empty one", async () => {
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") throw new Error("server-config.json is not valid JSON");
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    await waitFor(() =>
      expect(screen.getByText(/server-config.json is not valid JSON/)).toBeTruthy(),
    );
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });

  it("shows an empty state when nothing has been imported", async () => {
    show([model()]);
    expect(await screen.findByText("No imported models yet.")).toBeTruthy();
  });
});

describe("an owned job", () => {
  it("disables the actions that would start a second one", async () => {
    // Single-flight is Rust's rule. A button that stays enabled and answers with
    // "already running" makes the user discover the rule by being refused.
    show(
      [
        model({
          availability: "missing",
          quantization: {
            ...model().quantization,
            supports_prequantize: true,
            prequantize_components: components("transformer", "text_encoder", "vae"),
            prequantize_choices: [4],
            prequantize_strategy: "mflux_save",
          },
        }),
      ],
      idleJobs({ job: job({ state: "running", kind: "fetch", target: "flux2-dev" }), active: true }),
    );

    const it_ = await row("z-image-turbo");
    expect((it_.getByRole("button", { name: "Install" }) as HTMLButtonElement).disabled).toBe(true);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(
      (dialog.getByRole("button", { name: /pre-quantize/i }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("watches this model's conversion from the dialog it was started in", async () => {
    // Rust names the job `"{model} @ {bits}-bit"`. A conversion of another model
    // is not this one's to report, and is not reported here.
    show(
      [
        model({
          quantization: {
            ...model().quantization,
            supports_prequantize: true,
            prequantize_components: components("transformer", "text_encoder", "vae"),
            prequantize_choices: [4],
            prequantize_strategy: "mflux_save",
          },
        }),
      ],
      idleJobs({
        job: job({
          state: "running",
          kind: "prequantize",
          target: "z-image-turbo @ 4-bit",
          message: "converting",
        }),
        active: true,
      }),
    );

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.getByText("z-image-turbo @ 4-bit")).toBeTruthy();
    expect(dialog.getByText("converting")).toBeTruthy();
    expect(dialog.getByRole("button", { name: "Cancel" })).toBeTruthy();
  });

  it("does not show another model's conversion as this model's", async () => {
    show(
      [
        model({
          quantization: {
            ...model().quantization,
            supports_prequantize: true,
            prequantize_components: components("transformer", "text_encoder", "vae"),
            prequantize_choices: [4],
            prequantize_strategy: "mflux_save",
          },
        }),
      ],
      idleJobs({
        job: job({ state: "running", kind: "prequantize", target: "flux2-dev @ 4-bit" }),
        active: true,
      }),
    );

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.queryByRole("button", { name: "Cancel" })).toBeNull();
  });
});

describe("the enabled switch", () => {
  /** A backend whose `config_write` we can inspect. */
  function withConfig(models: ModelStatus[], config: unknown) {
    const writes: unknown[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "models_status") return catalogue(models);
      if (command === "config_write") {
        writes.push((args as { value?: unknown } | undefined)?.value);
        return null;
      }
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={config}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );
    return writes;
  }

  it("is a switch, not a badge, and reports its state", async () => {
    show([model()], idleJobs(), { models: { "z-image-turbo": { enabled: false } } });
    const it_ = await row("z-image-turbo");
    const control = it_.getByRole("switch", { name: "Enable z-image-turbo" });
    expect(control.getAttribute("aria-checked")).toBe("false");
  });

  it("writes the model's own `enabled` key and leaves every other key alone", async () => {
    const writes = withConfig([model()], {
      default_model: "z-image-turbo",
      server: { port: 8765 },
      models: { "z-image-turbo": { enabled: true, quantize: 4 }, "flux2-dev": { enabled: false } },
    });

    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("switch", { name: "Enable z-image-turbo" }));

    await waitFor(() => expect(writes).toHaveLength(1));
    const written = writes[0] as Record<string, any>;
    expect(written.models["z-image-turbo"].enabled).toBe(false);
    // The rest of the document survives: this is one key, not a rewrite.
    expect(written.models["z-image-turbo"].quantize).toBe(4);
    expect(written.models["flux2-dev"]).toEqual({ enabled: false });
    expect(written.server).toEqual({ port: 8765 });
    expect(written.default_model).toBe("z-image-turbo");
  });

  it("says a restart is needed while the running server still has the old set", async () => {
    // Two backend facts disagreeing: the configuration enables this model, and
    // `/v1/capabilities` — what the *running* server loaded — does not list it.
    const capabilities = {
      default_model: "z-image-turbo",
      max_n: 4,
      response_formats: ["url"],
      models: {},
    };
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue([model()]);
      throw new Error(`unexpected command ${command}`);
    });
    const client = { capabilities: async () => capabilities } as never;
    render(
      <Models
        state={overview({ server: { running: true, port: 8765, lastExit: null } })}
        client={client}
        config={{ models: { "z-image-turbo": { enabled: true } } }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await waitFor(() => expect(it_.getByText("restart required")).toBeTruthy());
  });

  it("says nothing about restarting when the server already agrees", async () => {
    const capabilities = {
      default_model: "z-image-turbo",
      max_n: 4,
      response_formats: ["url"],
      models: { "z-image-turbo": { active_variant: null } },
    };
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue([model()]);
      throw new Error(`unexpected command ${command}`);
    });
    const client = { capabilities: async () => capabilities } as never;
    render(
      <Models
        state={overview({ server: { running: true, port: 8765, lastExit: null } })}
        client={client}
        config={{ models: { "z-image-turbo": { enabled: true } } }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await waitFor(() => expect(it_.getByRole("switch")).toBeTruthy());
    expect(it_.queryByText("restart required")).toBeNull();
  });
});

describe("per-model settings", () => {
  it("opens the model's overrides in a dialog rather than expanding the row", async () => {
    show([model()], idleJobs(), { models: { "z-image-turbo": { default_steps: 12 } } });
    const it_ = await row("z-image-turbo");

    const open = it_.getByRole("button", { name: "Model settings…" });
    // Not a disclosure: nothing in the row expands, and nothing claims to.
    expect(open.getAttribute("aria-expanded")).toBeNull();
    expect(it_.queryByRole("dialog")).toBeNull();
    expect(it_.queryByLabelText("Steps for z-image-turbo")).toBeNull();

    await userEvent.click(open);
    const dialog = within(it_.getByRole("dialog"));
    expect((dialog.getByLabelText("Steps for z-image-turbo") as HTMLInputElement).value).toBe("12");
    // Quantization is the other dialog's subject, and is in neither this one nor
    // the row.
    expect(dialog.queryByLabelText(/quantization/i)).toBeNull();

    await userEvent.click(dialog.getByRole("button", { name: "Close" }));
    expect(it_.queryByRole("dialog")).toBeNull();
    // The row's own state was never hidden by any of it.
    expect(it_.getByRole("switch", { name: "Enable z-image-turbo" })).toBeTruthy();
  });

  it("closes on Escape, and hands focus back to what opened it", async () => {
    show([model()]);
    const it_ = await row("z-image-turbo");
    const open = it_.getByRole("button", { name: "Model settings…" });

    await userEvent.click(open);
    expect(it_.getByRole("dialog")).toBeTruthy();

    await userEvent.keyboard("{Escape}");
    expect(it_.queryByRole("dialog")).toBeNull();
    // A keyboard user is left where they were, not at the top of the document.
    expect(document.activeElement).toBe(open);
  });

  it("gives the edits toggle one accessible label on a full-width row", async () => {
    // It used to be a 14px checkbox in a 260px grid cell, which broke "Expose
    // the edits endpoint" over three lines beside it.
    show([model()]);

    const dialog = await openDialog("z-image-turbo", "Model settings…");
    const control = dialog.getByRole("switch", {
      name: "Expose the edits endpoint for z-image-turbo",
    });
    expect(control.getAttribute("aria-checked")).toBe("false");
    expect(dialog.queryByRole("checkbox")).toBeNull();

    // One label, and it is the control's own — not a caption beside it.
    const labels = dialog.getAllByText("Expose the edits endpoint");
    expect(labels).toHaveLength(1);
    expect(labels[0]!.parentElement?.className).toContain("switch-row");
    expect(labels[0]!.closest(".setting-pair")).toBeNull();
  });

  it("applies the whole draft in one write", async () => {
    const writes: unknown[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "models_status") return catalogue([model()]);
      if (command === "config_write") {
        writes.push((args as { value?: unknown } | undefined)?.value);
        return null;
      }
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{ models: {} }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("button", { name: /model settings/i }));
    await userEvent.clear(it_.getByLabelText("Steps for z-image-turbo"));
    await userEvent.type(it_.getByLabelText("Steps for z-image-turbo"), "20");
    await userEvent.click(it_.getByRole("button", { name: "Apply" }));

    // One write, not one per keystroke.
    await waitFor(() => expect(writes).toHaveLength(1));
    expect((writes[0] as Record<string, any>).models["z-image-turbo"].default_steps).toBe(20);
  });
});

describe("locating a built-in", () => {
  /** A backend that answers `local_model_locate` and records config writes. */
  function locateBackend(verdict: Record<string, unknown>, models = [model({ availability: "missing" })]) {
    const calls: { command: string; args?: unknown }[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      calls.push({ command, args });
      if (command === "models_status") return catalogue(models);
      if (command === "pick_directory") return "/Volumes/Big/models--mlx-community--Z-Image-bf16";
      if (command === "local_model_locate") return verdict;
      if (command === "config_write") return null;
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{ models: {} }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );
    return calls;
  }

  const ok = {
    ok: true,
    path: "/Volumes/Big/models--mlx-community--Z-Image-bf16",
    model: "z-image-turbo",
    availability: "present",
    family: "z-image",
    class_name: "ZImageTransformer2DModel",
    reason: null,
    detected_repo: "mlx-community/Z-Image-bf16",
    repo_verified: true,
  };

  it("offers Locate beside Install for a model that is not installed", async () => {
    locateBackend(ok);
    const it_ = await row("z-image-turbo");
    expect(it_.getByRole("button", { name: "Install" })).toBeTruthy();
    expect(it_.getByRole("button", { name: "Locate…" })).toBeTruthy();
  });

  it("binds the chosen folder through the model's own config override", async () => {
    const calls = locateBackend(ok);
    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("button", { name: "Locate…" }));

    // Checked, and confirmed by the user — never bound straight from the picker.
    await waitFor(() => expect(screen.getByRole("button", { name: "Use this folder" })).toBeTruthy());
    expect(calls.some((c) => c.command === "config_write")).toBe(false);

    await userEvent.click(screen.getByRole("button", { name: "Use this folder" }));
    await waitFor(() => expect(calls.some((c) => c.command === "config_write")).toBe(true));

    const write = calls.find((c) => c.command === "config_write")!.args as {
      value: Record<string, any>;
    };
    expect(write.value.models["z-image-turbo"].model_path).toBe(ok.path);
    // The catalogue identity is untouched: no new entry, no renaming.
    expect(Object.keys(write.value.models)).toEqual(["z-image-turbo"]);
  });

  it("says so when the repository identity is proven", async () => {
    locateBackend(ok);
    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("button", { name: "Locate…" }));
    expect(await screen.findByText(/its identity is confirmed/)).toBeTruthy();
  });

  it("does not claim provenance it cannot prove", async () => {
    // Compatible is not "this is that repository", and the difference is stated
    // before the binding rather than smoothed over afterwards.
    locateBackend({ ...ok, detected_repo: null, repo_verified: false });
    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("button", { name: "Locate…" }));
    expect(await screen.findByText(/but not that it is the exact repository/)).toBeTruthy();
    expect(screen.queryByText(/identity is confirmed/)).toBeNull();
  });

  it("surfaces an incompatible family instead of binding it", async () => {
    locateBackend({
      ...ok,
      ok: false,
      availability: "incompatible",
      reason: "This directory holds a 'z-image' model, but 'flux2-klein' is 'flux2'.",
    });
    const it_ = await row("z-image-turbo");
    await userEvent.click(it_.getByRole("button", { name: "Locate…" }));
    await waitFor(() =>
      expect(screen.getByText(/but 'flux2-klein' is 'flux2'/)).toBeTruthy(),
    );
    expect(screen.queryByRole("button", { name: "Use this folder" })).toBeNull();
  });

  it("offers Reset location once a built-in reads from a local folder, and no Install", async () => {
    // `can_download` false is what removes Install: the backend decides, and a
    // located built-in is no longer a download target.
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status")
        return catalogue([
          model({ availability: "present", local: true, can_download: false, repo: "/Volumes/Big/z" }),
        ]);
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{ models: { "z-image-turbo": { model_path: "/Volumes/Big/z" } } }}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const it_ = await row("z-image-turbo");
    expect(it_.getByRole("button", { name: "Reset location" })).toBeTruthy();
    expect(it_.queryByRole("button", { name: /install|resume|locate/i })).toBeNull();
    // Still a built-in, still in the catalogue list — not an imported row.
    expect(it_.queryByText("imported local")).toBeNull();
    expect(screen.getByText("No imported models yet.")).toBeTruthy();
  });
});

describe("public API identity", () => {
  it("shows an imported model's API name, not its internal id", async () => {
    show([
      model({
        key: "local-c1587aa663c4",
        display_name: "My Z-Image",
        api_name: "my-z-image",
        provenance: "imported_local",
        can_download: false,
      }),
    ]);

    const it_ = await row("My Z-Image");
    expect(it_.getByText("my-z-image")).toBeTruthy();
    // The opaque id is storage, and is not offered as the thing to send.
    expect(it_.queryByText("local-c1587aa663c4")).toBeNull();
  });

  it("defaults the API name from the display name, and stops once edited", async () => {
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return catalogue([]);
      if (command === "pick_directory") return "/models/z";
      if (command === "local_model_inspect")
        return {
          ok: true,
          path: "/models/z",
          availability: "present",
          family: "z-image",
          class_name: "ZImageTransformer2DModel",
          suggested_name: "z",
          reason: null,
          profiles: ["z-image-turbo"],
          suggested_api_name: "z",
        };
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    await userEvent.click(await screen.findByRole("button", { name: /Import Local Model/ }));
    const apiName = (await screen.findByLabelText("API name")) as HTMLInputElement;
    expect(apiName.value).toBe("z");

    const display = screen.getByLabelText("Display name");
    await userEvent.clear(display);
    await userEvent.type(display, "My Z-Image");
    expect(apiName.value).toBe("my-z-image");

    // Edited on its own, it stops following.
    await userEvent.clear(apiName);
    await userEvent.type(apiName, "custom");
    await userEvent.type(display, "!");
    expect(apiName.value).toBe("custom");
  });
});

describe("components", () => {
  /** A model whose components and partial progress come from the backend. */
  const convertible = (patch: Partial<ModelStatus> = {}) =>
    model({
      quantization: {
        ...model().quantization,
        supports_prequantize: true,
        prequantize_choices: [4],
        prequantize_strategy: "qds_memory_bounded",
        // Deliberately not the real catalogue's three: the interface renders
        // what it is handed, and a test that used the real names could not tell
        // a rendered list from a remembered one.
        prequantize_components: components("transformer", "adapter"),
      },
      ...patch,
    });

  it("renders the components the backend published, and no others", async () => {
    show([convertible()]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const list = dialog.getByRole("list", { name: "Components" });
    expect(within(list).getAllByRole("checkbox").map((box) => box.getAttribute("aria-label") ?? "")).toHaveLength(2);
    expect(within(list).getByText("Transformer")).toBeTruthy();
    expect(within(list).getByText("Adapter")).toBeTruthy();
    // The FLUX.2-dev names this file used to carry are not a property of every
    // model, and this model does not have them.
    expect(within(list).queryByText("VAE")).toBeNull();
    expect(within(list).queryByText("Text encoder")).toBeNull();
  });

  it("keeps each row's box, name and status together and in that order", async () => {
    // The row used to centre the name between a left-hand box and a right-hand
    // pill, which wrapped "Text encoder" over two lines with space at both ends.
    show([
      convertible({
        quantization: {
          ...convertible().quantization,
          prequantize_components: [
            {
              key: "text_encoder",
              label: "Text encoder",
              required: true,
              independently_convertible: true,
              quantized: false,
              note: "mflux does not quantize this encoder.",
            },
          ],
        },
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const [entry] = dialog.getAllByRole("listitem");

    // One control, named by the text beside it: clicking the name toggles it.
    const box = within(entry!).getByRole("checkbox", { name: /Text encoder/ });
    const label = box.closest("label") as HTMLElement;
    expect(label).not.toBeNull();
    expect(within(label).getByText("Text encoder")).toBeTruthy();
    // The note travels with the identity rather than displacing it.
    expect(within(label).getByText("saved at source precision")).toBeTruthy();

    // The status is the row's, not the label's: it is what the free space sits
    // before, so every row's pill lines up.
    const status = within(entry!).getByText("Not converted");
    expect(label.contains(status)).toBe(false);
    expect(status.parentElement).toBe(entry);
  });

  it("shows which components are already converted at the chosen depth", async () => {
    show([
      convertible({
        partials: [
          {
            bits: 4,
            path: "/a/4bit",
            strategy: "qds_memory_bounded",
            components: { transformer: "complete", adapter: "missing" },
            size_bytes: 3_000_000_000,
          },
        ],
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    const rows = dialog.getAllByRole("listitem");
    expect(within(rows[0]!).getByText("Converted")).toBeTruthy();
    expect(within(rows[1]!).getByText("Not converted")).toBeTruthy();
  });

  it("offers to continue rather than to start, and says what is still missing", async () => {
    show([
      convertible({
        partials: [
          {
            bits: 4,
            path: "/a/4bit",
            strategy: "qds_memory_bounded",
            components: { transformer: "complete", adapter: "missing" },
            size_bytes: 3_000_000_000,
          },
        ],
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.getByRole("button", { name: "Continue pre-quantization" })).toBeTruthy();
    // And it must not read as usable: a partial copy is not a variant.
    expect(dialog.getByText(/cannot be used until/i)).toBeTruthy();
    expect(dialog.queryByRole("button", { name: /Use 4-bit/ })).toBeNull();
  });

  it("sends exactly the components that were selected", async () => {
    const calls: { command: string; args?: unknown }[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      calls.push({ command, args });
      if (command === "models_status") return catalogue([convertible()]);
      if (command === "prequantize_run") return null;
      throw new Error(`unexpected command ${command}`);
    });
    render(
      <Models
        state={overview()}
        client={null}
        config={{}}
        jobs={idleJobs()}
        onConfigChanged={onConfigChanged}
      />,
    );

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    // Nothing converted yet, so both are selected; drop one.
    await userEvent.click(dialog.getByRole("checkbox", { name: "Adapter" }));
    await userEvent.click(dialog.getByRole("button", { name: "Pre-quantize selected" }));

    await waitFor(() => expect(calls.some((c) => c.command === "prequantize_run")).toBe(true));
    const run = calls.find((c) => c.command === "prequantize_run")!.args as {
      components: string[];
      bits: number;
    };
    expect(run.components).toEqual(["transformer"]);
    expect(run.bits).toBe(4);
  });

  it("offers no component controls for a family whose components are unestablished", async () => {
    // Fails closed: an empty published list is not an invitation to guess three.
    show([
      convertible({
        quantization: {
          ...convertible().quantization,
          prequantize_components: [],
        },
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.queryByRole("list", { name: "Components" })).toBeNull();
  });

  it("treats a complete variant as every component converted", async () => {
    show([
      convertible({
        variants: [
          {
            bits: 4,
            path: "/a/4bit",
            strategy: "qds_memory_bounded",
            legacy: false,
            size_bytes: 5_900_000_000,
          },
        ],
      }),
    ]);

    const dialog = await openDialog("z-image-turbo", "Quantization…");
    expect(dialog.getAllByText("Converted")).toHaveLength(2);
    expect(dialog.queryByText(/cannot be used until/i)).toBeNull();
    expect(dialog.getByRole("button", { name: /Use 4-bit/ })).toBeTruthy();
  });
});

describe("disk usage", () => {
  it("shows the active representation's size, not the source's", async () => {
    show([
      model({
        active_variant: 4,
        variants: [
          { bits: 4, path: "/a/4bit", strategy: null, legacy: false, size_bytes: 5_900_000_000 },
        ],
        disk: disk({
          source_bytes: 20_500_000_000,
          active_bytes: 5_900_000_000,
          total_bytes: 26_400_000_000,
          breakdown: [
            { kind: "source", bits: null, bytes: 20_500_000_000, path: "/hf", is_source: true },
            { kind: "variant", bits: 4, bytes: 5_900_000_000, path: "/a/4bit", is_source: false },
          ],
        }),
      }),
    ]);

    const it_ = await row("z-image-turbo");
    // The badge itself, not one of the breakdown's lines.
    const badge = it_.getByTitle("What this model occupies on disk");
    expect(badge.textContent).toContain("5.90 GB");
    expect(badge.textContent).toContain("4-bit");
    // The source figure belongs to the breakdown, never to the badge.
    expect(badge.textContent).not.toContain("20.5 GB");
  });

  it("breaks the total down without hover, and counts a shared directory once", async () => {
    // FLUX.2-dev's source *is* its 8-bit artifact: one directory, one line, and
    // a total that is not twice its size.
    show([
      model({
        key: "flux2-dev",
        display_name: "FLUX.2-dev",
        disk: disk({
          source_bytes: 58_700_000_000,
          active_bytes: 58_700_000_000,
          total_bytes: 58_700_000_000,
          breakdown: [
            { kind: "variant", bits: 8, bytes: 58_700_000_000, path: "/a/8bit", is_source: true },
            { kind: "partial", bits: 4, bytes: 3_000_000_000, path: "/a/4bit", is_source: false },
          ],
        }),
      }),
    ]);

    const it_ = await row("FLUX.2-dev");
    // A disclosure, so it is reachable by keyboard rather than by pointer only.
    const summary = it_.getByTitle("What this model occupies on disk");
    expect(summary.tagName).toBe("SUMMARY");
    await userEvent.click(summary);

    const group = it_.getByRole("group", { name: "Disk usage" });
    expect(within(group).getByText(/8-bit artifact \(this model's source\)/)).toBeTruthy();
    expect(within(group).getByText(/4-bit conversion in progress/)).toBeTruthy();
    expect(within(group).getByText("Total")).toBeTruthy();
    // Two entries, one of which is the source — and one total, not their sum.
    expect(within(group).getAllByText("58.7 GB")).toHaveLength(2);
  });

  it("says nothing at all when no size is known", async () => {
    // A model that is not on this machine has no disk usage, and its catalogue
    // size is what a download would cost — a different question, labelled as one.
    show([
      model({
        availability: "missing",
        size_gb: 20.5,
        disk: disk({ source_bytes: null, active_bytes: null, total_bytes: 0, breakdown: [] }),
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByRole("group", { name: "Disk usage" })).toBeNull();
    expect(it_.getByText("20.5 GB to download")).toBeTruthy();
  });
});

describe("Hugging Face access", () => {
  it("does not edit the token: that is one global secret, and it lives in Configuration", async () => {
    show([model({ key: "flux2-dev", display_name: "FLUX.2-dev", gated: true })]);
    await screen.findByText("FLUX.2-dev");

    expect(screen.queryByLabelText(/Hugging Face token/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /save token/i })).toBeNull();
    expect(screen.queryByText(/HuggingFace access/i)).toBeNull();
  });

  it("says what a missing token costs, and where to fix it", async () => {
    show(
      [
        model({
          key: "flux2-dev",
          display_name: "FLUX.2-dev",
          gated: true,
          availability: "missing",
        }),
      ],
      idleJobs(),
      {},
      overview({ hfTokenPresent: false }),
    );

    await screen.findByText("FLUX.2-dev");
    expect(screen.getByText("Configuration → Hugging Face")).toBeTruthy();
    // And the download button says so before it is pressed.
    const it_ = await row("FLUX.2-dev");
    expect(it_.getByRole("button", { name: /Install/ }).getAttribute("title")).toMatch(
      /gated.*Configuration/,
    );
  });

  it("says nothing about tokens when one is present", async () => {
    show([model({ key: "flux2-dev", display_name: "FLUX.2-dev", gated: true })]);
    await screen.findByText("FLUX.2-dev");
    expect(screen.queryByText("Configuration → Hugging Face")).toBeNull();
  });
});
