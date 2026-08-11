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
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Models } from "./panels/Models";
import type { JobView } from "./job";
import { job, model, overview } from "./test-fixtures";
import type { ModelStatus } from "./types";

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

function show(models: ModelStatus[], jobs: JobView = idleJobs(), config: unknown = {}) {
  mockInvoke.mockImplementation(async (command: string) => {
    if (command === "models_status") return models;
    if (command === "local_model_forget") return { ok: true };
    throw new Error(`unexpected command ${command}`);
  });
  return render(
    <Models
      state={overview()}
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
    expect(it_.queryByRole("button", { name: /pre-quantize/i })).toBeNull();
    expect(it_.queryByLabelText(/bit depth/i)).toBeNull();
  });
});

describe("quantization", () => {
  it("offers exactly the bit depths the backend published", async () => {
    // Not a list of its own: 5 and 8 are absent here and must stay absent.
    show([
      model({
        quantization: {
          ...model().quantization,
          supports_prequantize: true,
          prequantize_choices: [3, 4, 6],
          prequantize_strategy: "mflux_save",
        },
      }),
    ]);

    const it_ = await row("z-image-turbo");
    const select = it_.getByLabelText("Bit depth for z-image-turbo") as HTMLSelectElement;
    expect([...select.options].map((option) => option.value)).toEqual(["3", "4", "6"]);
  });

  it("marks which saved variant is active, and offers a way back to the source", async () => {
    // Source, saved variants and the active representation are three different
    // facts. The 4-bit variant exists *and* is in use; the 3-bit only exists.
    show([
      model({
        quantization: {
          ...model().quantization,
          supports_prequantize: true,
          prequantize_choices: [3, 4],
          prequantize_strategy: "mflux_save",
        },
        variants: [
          { bits: 3, path: "/a/3bit", strategy: "mflux_save", legacy: false },
          { bits: 4, path: "/a/4bit", strategy: "mflux_save", legacy: false },
        ],
        active_variant: 4,
      }),
    ]);

    const it_ = await row("z-image-turbo");
    const active = it_.getByRole("button", { name: /4-bit · active/ });
    expect(active.getAttribute("aria-pressed")).toBe("true");
    expect((active as HTMLButtonElement).disabled).toBe(true);

    const inactive = it_.getByRole("button", { name: "3-bit" });
    expect(inactive.getAttribute("aria-pressed")).toBe("false");
    expect((inactive as HTMLButtonElement).disabled).toBe(false);

    expect(it_.getByRole("button", { name: "Use original" })).toBeTruthy();
  });

  it("claims no variant for a model that has none, but still offers to make one", async () => {
    show([
      model({
        quantization: {
          ...model().quantization,
          supports_prequantize: true,
          prequantize_choices: [4],
          prequantize_strategy: "mflux_save",
        },
      }),
    ]);

    const it_ = await row("z-image-turbo");
    expect(it_.queryByText(/saved variants/i)).toBeNull();
    expect(it_.queryByRole("button", { name: "Use original" })).toBeNull();
    // The affordance is what says a variant *could* exist.
    expect(it_.getByRole("button", { name: /pre-quantize/i })).toBeTruthy();
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
        return [
          model({
            key: "local-abc",
            display_name: "My local model",
            provenance: "imported_local",
            can_download: false,
          }),
        ];
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
            prequantize_choices: [4],
            prequantize_strategy: "mflux_save",
          },
        }),
      ],
      idleJobs({ job: job({ state: "running", kind: "fetch", target: "flux2-dev" }), active: true }),
    );

    const it_ = await row("z-image-turbo");
    expect((it_.getByRole("button", { name: "Install" }) as HTMLButtonElement).disabled).toBe(true);
    expect(
      (it_.getByRole("button", { name: /pre-quantize/i }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("the enabled switch", () => {
  /** A backend whose `config_write` we can inspect. */
  function withConfig(models: ModelStatus[], config: unknown) {
    const writes: unknown[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "models_status") return models;
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
      if (command === "models_status") return [model()];
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
      models: { "z-image-turbo": {} },
    };
    mockInvoke.mockImplementation(async (command: string) => {
      if (command === "models_status") return [model()];
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
  it("keeps the model's own overrides in its row, behind a disclosure", async () => {
    show([model()], idleJobs(), { models: { "z-image-turbo": { default_steps: 12 } } });
    const it_ = await row("z-image-turbo");

    const toggle = it_.getByRole("button", { name: /model settings/i });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // Core state stays visible whether or not the settings are open.
    expect(it_.queryByLabelText("Steps for z-image-turbo")).toBeNull();

    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const steps = it_.getByLabelText("Steps for z-image-turbo") as HTMLInputElement;
    expect(steps.value).toBe("12");
    expect(it_.getByRole("switch", { name: "Enable z-image-turbo" })).toBeTruthy();
  });

  it("applies the whole draft in one write", async () => {
    const writes: unknown[] = [];
    mockInvoke.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "models_status") return [model()];
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
      if (command === "models_status") return models;
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
        return [model({ availability: "present", local: true, can_download: false, repo: "/Volumes/Big/z" })];
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
      if (command === "models_status") return [];
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
