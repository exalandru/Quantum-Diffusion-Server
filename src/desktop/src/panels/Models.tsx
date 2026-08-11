import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import { ActionNote, useActions } from "../actions";
import { describeJob, type JobView } from "../job";
import type {
  Availability,
  Capabilities,
  ImportVerdict,
  ModelCapabilities,
  LocateVerdict,
  ModelStatus,
  Overview,
} from "../types";

/**
 * How each availability is *presented*, and what it must not offer.
 *
 * A label-and-tone map, not a rule: whether the Install button may appear at all
 * is `can_download`, which the backend decides. The one thing worth restating is
 * why `volume_unmounted` and `unreadable` carry no label — the weights are not
 * gone, the disk holding them is unplugged or unreadable, and offering "Install"
 * there would invite a re-download of tens of gigabytes the user already owns.
 */
const ACTION: Record<Availability, { label: string | null; badge: string; tone: string }> = {
  present: { label: null, badge: "installed", tone: "pill-ok" },
  partial: { label: "Resume", badge: "incomplete", tone: "pill-warn" },
  missing: { label: "Install", badge: "not installed", tone: "pill" },
  volume_unmounted: { label: null, badge: "volume unavailable", tone: "pill-warn" },
  unreadable: { label: null, badge: "unreadable", tone: "pill-bad" },
};

/**
 * Order enforced by the conversion script: biggest first, to bound the disk peak.
 *
 * The one piece of model knowledge left in this file, and it predates the
 * redesign. It is FLUX.2-dev's component list for the memory-bounded strategy;
 * moving it behind a backend field is a real change to a published contract, so
 * it stays where it was rather than being smuggled into a visual slice.
 */
const COMPONENTS = [
  { id: "transformer", label: "Transformer", detail: "64.5 GB in bf16 → about 34 GB" },
  { id: "text_encoder", label: "Text encoder", detail: "45.8 GB in bf16 → about 24 GB" },
  { id: "vae", label: "VAE", detail: "0.34 GB" },
];

export function Models({
  state,
  client,
  config,
  jobs,
  onConfigChanged,
}: {
  state: Overview;
  client: ServerClient | null;
  config: unknown;
  jobs: JobView;
  /** Re-read the configuration after this view writes to it. */
  onConfigChanged: () => void | Promise<void>;
}) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [capabilitiesNote, setCapabilitiesNote] = useState<string | null>(null);
  const [models, setModels] = useState<ModelStatus[] | null>(null);
  // Reading the catalogue is a panel load, not a user action, so it gets its own
  // slot: it used to share the global error, which meant a broken
  // `server-config.json` blanked this view and then had its explanation wiped by
  // the next status poll.
  const [catalogueError, setCatalogueError] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [selected, setSelected] = useState<string[]>(COMPONENTS.map((component) => component.id));
  /** Chosen bit depth per model. Seeded from the backend's published choices. */
  const [bitsFor, setBitsFor] = useState<Record<string, number>>({});
  /** Pending import awaiting confirmation. Inspection is advisory only —
      registration revalidates on the Python side. */
  const [pending, setPending] = useState<ImportVerdict | null>(null);
  const [pendingProfile, setPendingProfile] = useState<string>("");
  const [pendingName, setPendingName] = useState<string>("");
  const [pendingApiName, setPendingApiName] = useState<string>("");
  /** A checked-but-unbound Locate awaiting confirmation. */
  const [locating, setLocating] = useState<LocateVerdict | null>(null);
  const { run, dismiss, stateOf, busy, anyBusy } = useActions();
  /** Terminal job result the user has dismissed, keyed by when it finished. */
  const [dismissedAt, setDismissedAt] = useState<number | null>(null);
  /** Once the API name is typed in, it stops following the display name. */
  const apiNameEdited = useRef(false);

  useEffect(() => {
    if (!client) {
      setCapabilitiesNote(null);
      return;
    }
    void client
      .capabilities()
      .then((value) => {
        setCapabilities(value);
        setCapabilitiesNote(null);
      })
      .catch((cause) => {
        setCapabilities(null);
        setCapabilitiesNote(messageOf(cause));
      });
  }, [client]);

  // The catalogue comes from Rust, not from the server: that way it lists the
  // disabled models too, and it works with the server stopped — which is when you
  // most want to download weights.
  const reloadModels = useCallback(async () => {
    try {
      setModels(await api.modelsStatus());
      setCatalogueError(null);
    } catch (cause) {
      setCatalogueError(messageOf(cause));
    }
  }, []);

  useEffect(() => {
    void reloadModels();
  }, [reloadModels]);

  // A finished download changes what is on disk, so the catalogue is stale.
  useEffect(
    () =>
      jobs.onSettled((settled) => {
        if (settled.kind === "fetch" && settled.state === "completed") void reloadModels();
      }),
    [jobs, reloadModels],
  );

  // The action covers only *starting* the job — that is what can be refused
  // (single-flight) or fail to spawn. How it ends is Rust's to report, and
  // survives this panel unmounting.
  async function download(key: string) {
    if (await run(`download:${key}`, () => api.modelFetch(key))) {
      setDismissedAt(null);
      await jobs.refresh();
    }
  }

  async function chooseImport() {
    await run("import", async () => {
      const path = await api.pickDirectory(null);
      if (!path) return; // cancelling is not a failure
      const verdict = await api.localModelInspect(path);
      if (!verdict.ok) throw new Error(verdict.reason ?? `Cannot import: ${verdict.availability}`);
      if (verdict.already_imported) {
        throw new Error(`Already imported as "${verdict.already_imported.display_name}".`);
      }
      setPending(verdict);
      // Preselected only when there is nothing to choose between.
      setPendingProfile(verdict.profiles.length === 1 ? verdict.profiles[0]! : "");
      setPendingName(verdict.suggested_name ?? "");
      setPendingApiName(verdict.suggested_api_name ?? "");
      apiNameEdited.current = false;
    });
  }

  /**
   * Point a built-in catalogue entry at weights already on this machine.
   *
   * Distinct from importing, and the difference is the identity: this leaves the
   * catalogue key, family and profile exactly as shipped and only tells the entry
   * where its weights are. Importing mints a new entry the user owns.
   */
  async function chooseLocation(model: ModelStatus) {
    await run(`locate:${model.key}`, async () => {
      const path = await api.pickDirectory(state.hfHome);
      if (!path) return; // cancelling is not a failure
      const verdict = await api.localModelLocate(path, model.key);
      if (!verdict.ok) throw new Error(verdict.reason ?? "This directory cannot be used here.");
      setLocating(verdict);
    });
  }

  /** Bind the checked directory, through the same config path as every override. */
  async function confirmLocation() {
    if (!locating) return;
    const model = (models ?? []).find((row) => row.key === locating.model);
    if (!model) return;
    await patchModel(
      model,
      { model_path: locating.path },
      `locate:${model.key}`,
      "Located. The catalogue entry now reads from that folder.",
    );
    setLocating(null);
  }

  async function confirmImport() {
    if (!pending || !pendingProfile) return;
    await run(
      "import",
      async () => {
        const result = await api.localModelRegister(
          pending.path,
          pendingProfile,
          pendingName,
          pendingApiName,
        );
        if (!result.ok) throw new Error(result.reason ?? "Registration failed.");
        setPending(null);
        await reloadModels();
      },
      "Model imported.",
    );
  }

  async function forget(model: ModelStatus) {
    await run(
      `forget:${model.key}`,
      async () => {
        const result = await api.localModelForget(model.key);
        // The backend refuses when this model is the configured default, and its
        // sentence names the fix. Reworded here it would stop being the fix.
        if (!result.ok) throw new Error(result.reason ?? "Could not forget this model.");
        await reloadModels();
      },
      "Removed from the library. The model files were not touched.",
    );
  }

  async function convert(model: ModelStatus, bits: number) {
    // Component selection is meaningful only to the memory-bounded strategy,
    // which works through FLUX.2-dev one component at a time. `mflux_save` writes
    // the whole model in one call, so sending a subset would be a lie.
    const components =
      model.quantization.prequantize_strategy === "qds_memory_bounded" ? selected : [];
    if (await run(`convert:${model.key}`, () => api.prequantizeRun(model.key, bits, components))) {
      setDismissedAt(null);
      await jobs.refresh();
    }
  }

  /**
   * Apply an override patch to one model, through the existing config path.
   *
   * The whole document is rewritten because that is what `config_write` takes;
   * only this model's entry changes. Validation stays where it is enforced — the
   * server rejects a value it does not accept, and this does not second-guess it.
   */
  async function patchModel(model: ModelStatus, patch: Record<string, unknown>, key: string, success?: string) {
    await run(
      key,
      async () => {
        const next = structuredClone(config ?? {}) as Record<string, any>;
        next.models = { ...(next.models ?? {}) };
        next.models[model.key] = { ...(next.models[model.key] ?? {}), ...patch };
        await api.configWrite(next);
        await onConfigChanged();
        await reloadModels();
      },
      success,
    );
  }

  /** Activation is explicit and separate: it never touches `model_path`. */
  async function activate(model: ModelStatus, bits: number | null) {
    await patchModel(
      model,
      { prequantized_variant: bits },
      `activate:${model.key}`,
      bits === null ? "Using the original model." : `Using the ${bits}-bit variant.`,
    );
  }

  /** Drop a local override and go back to the catalogue's repository. */
  async function resetLocation(model: ModelStatus) {
    await patchModel(
      model,
      { model_path: null },
      `locate:${model.key}`,
      "Location reset. The files were left where they are.",
    );
  }

  async function saveToken() {
    await run(
      "token",
      async () => {
        await api.hfTokenWrite(token);
        setToken("");
      },
      "Token saved.",
    );
  }

  const job = jobs.job;
  // Rust settled a terminal outcome the user has not dismissed. Shown through the
  // same note as an action result, but sourced from Rust — polling re-reads it
  // rather than clearing it.
  const terminal =
    job && !jobs.active && job.state !== "idle" && job.finishedAtMs !== dismissedAt ? job : null;

  const builtIn = (models ?? []).filter((model) => model.provenance === "built_in");
  const imported = (models ?? []).filter((model) => model.provenance === "imported_local");
  const gatedRepos = builtIn.filter((model) => model.gated).map((model) => model.repo);

  const overrideOf = (key: string): Record<string, any> => {
    const models = (config as { models?: Record<string, any> } | null)?.models;
    return models?.[key] ?? {};
  };

  // What the *running* server exposes, which is not what the configuration says
  // until it has been restarted. `null` when there is no answer to compare with.
  const servedBy = (key: string): boolean | null =>
    capabilities ? key in capabilities.models : null;

  const rowProps = (model: ModelStatus) => ({
    model,
    caps: capabilities?.models[model.key],
    isDefault: model.key === capabilities?.default_model,
    tokenPresent: state.hfTokenPresent,
    anyBusy,
    jobActive: jobs.active,
    busy,
    stateOf,
    dismiss,
    override: overrideOf(model.key),
    servedByServer: servedBy(model.key),
    bits: bitsFor[model.key] ?? lastChoice(model),
    onBits: (value: number) => setBitsFor((previous) => ({ ...previous, [model.key]: value })),
    selectedComponents: selected,
    onComponents: setSelected,
    onDownload: () => void download(model.key),
    onConvert: (value: number) => void convert(model, value),
    onActivate: (value: number | null) => void activate(model, value),
    onForget: () => void forget(model),
    located: typeof overrideOf(model.key).model_path === "string",
    onLocate: () => void chooseLocation(model),
    onResetLocation: () => void resetLocation(model),
    onPatch: (patch: Record<string, unknown>, key: string, success?: string) =>
      void patchModel(model, patch, key, success),
  });

  return (
    <>
      {/* ── The Rust-owned operation ────────────────────────────────────── */}
      {(jobs.active || terminal) && job && (
        <section className="panel">
          <div className="row spread">
            <div className="row">
              <h2 style={{ margin: 0 }}>
                {job.kind === "prequantize" ? "Conversion" : "Download"}
              </h2>
              {job.target && <strong>{job.target}</strong>}
              {jobs.active && (
                <span className={job.state === "cancelling" ? "pill pill-warn" : "pill pill-live"}>
                  {job.state === "cancelling" ? "Stopping" : "Running"}
                </span>
              )}
            </div>
            {jobs.active && (
              <button
                className="danger"
                onClick={() => void run("cancel", async () => void (await api.jobCancel()))}
                disabled={job.state === "cancelling" || busy("cancel")}
              >
                {job.state === "cancelling" ? "Stopping…" : "Cancel"}
              </button>
            )}
          </div>

          {jobs.active && (
            <>
              <p className="note" style={{ marginBottom: 8 }}>
                {describeJob(job)}
              </p>
              {/* The child reports phases and blocks, never a byte total, so an
                  indeterminate stripe is the honest rendering. */}
              <div className="bar bar-indeterminate" />
            </>
          )}

          {terminal && (
            <ActionNote
              state={
                terminal.state === "completed"
                  ? { status: "ok", message: `${terminal.target ?? "Operation"} finished.` }
                  : {
                      status: "error",
                      message: `${terminal.target ?? "Operation"} ${terminal.state}: ${
                        terminal.message ?? "no reason reported"
                      }`,
                    }
              }
              onDismiss={() => setDismissedAt(terminal.finishedAtMs)}
            />
          )}
          <ActionNote state={stateOf("cancel")} onDismiss={() => dismiss("cancel")} />
        </section>
      )}

      {/* ── Built-in catalogue ──────────────────────────────────────────── */}
      <section className="panel">
        <h2>Catalogue</h2>
        <p className="note">
          Weights are downloaded on demand: the first generation on a fresh model would otherwise
          silently pay tens of gigabytes. Generation defaults come from the server and appear once it
          is running.
        </p>

        {locating && (
          <LocateConfirmation
            verdict={locating}
            busy={busy(`locate:${locating.model}`)}
            onConfirm={() => void confirmLocation()}
            onCancel={() => setLocating(null)}
          />
        )}

        {catalogueError ? (
          <div className="notice notice-error" style={{ marginTop: 12, marginBottom: 0 }}>
            <div className="row spread">
              <span>
                <strong>The catalogue could not be read.</strong> {catalogueError}
              </span>
              <button className="small" onClick={() => void reloadModels()}>
                Retry
              </button>
            </div>
          </div>
        ) : !models ? (
          <p className="empty" style={{ marginTop: 12 }}>
            Reading the catalogue…
          </p>
        ) : (
          <ul className="models" aria-label="Built-in models">
            {builtIn.map((model) => (
              <ModelRow key={model.key} {...rowProps(model)} />
            ))}
          </ul>
        )}

        {models && !capabilities && (
          <p className="note">
            {capabilitiesNote
              ? `The server is running but did not answer /v1/capabilities: ${capabilitiesNote}`
              : "Start the server to fill in each model's generation defaults."}
          </p>
        )}
      </section>

      {/* ── Imported local models ───────────────────────────────────────── */}
      <section className="panel">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Local models</h2>
          <button onClick={() => void chooseImport()} disabled={busy("import")}>
            {busy("import") ? "Working…" : "Import Local Model…"}
          </button>
        </div>
        <p className="note">
          Register a model already on this machine. Nothing is copied — QDS records where it is, and
          Forget removes only that record.
        </p>
        <ActionNote state={stateOf("import")} onDismiss={() => dismiss("import")} />

        {pending && (
          <ImportConfirmation
            verdict={pending}
            profile={pendingProfile}
            name={pendingName}
            apiName={pendingApiName}
            capabilities={capabilities}
            busy={busy("import")}
            onProfile={setPendingProfile}
            onName={(value) => {
              setPendingName(value);
              // The public name follows the display name until it is edited on
              // its own: an alias nobody chose should still track what they typed.
              if (!apiNameEdited.current) setPendingApiName(slugFor(value));
            }}
            onApiName={(value) => {
              apiNameEdited.current = true;
              setPendingApiName(value);
            }}
            onConfirm={() => void confirmImport()}
            onCancel={() => setPending(null)}
          />
        )}

        {models &&
          (imported.length === 0 ? (
            <p className="empty" style={{ marginTop: 12 }}>
              No imported models yet.
            </p>
          ) : (
            <ul className="models" aria-label="Imported local models">
              {imported.map((model) => (
                <ModelRow key={model.key} {...rowProps(model)} />
              ))}
            </ul>
          ))}
      </section>

      <hr className="divider" />

      {/* ── HuggingFace access ──────────────────────────────────────────── */}
      <section className="panel">
        <h2>HuggingFace access</h2>
        <p className="note">
          {gatedRepos.length > 0 && (
            <>
              {gatedRepos.length} catalogue repositories are gated — the{" "}
              {[...new Set(gatedRepos.map((repo) => repo.split("/")[0]))].map((org, index) => (
                <span key={org}>
                  {index > 0 && ", "}
                  <code>{org}/*</code>
                </span>
              ))}{" "}
              ones.{" "}
            </>
          )}
          A gated repository needs a token granted access on its model card. It is stored where{" "}
          <code>hf auth login</code> writes it, so as not to duplicate a secret that already sits
          there in plaintext.
        </p>
        <div className="row" style={{ marginTop: 12 }}>
          <span className={state.hfTokenPresent ? "pill pill-ok" : "pill pill-warn"}>
            {state.hfTokenPresent ? "token present" : "no token"}
          </span>
          <input
            type="password"
            aria-label="HuggingFace token"
            placeholder="hf_…"
            value={token}
            onChange={(event) => {
              setToken(event.target.value);
              // Typing a new token supersedes the previous save's result.
              dismiss("token");
            }}
            style={{ flex: 1, minWidth: 220 }}
          />
          <button
            onClick={() => void saveToken()}
            disabled={token.trim().length === 0 || busy("token")}
          >
            {busy("token") ? "Saving…" : "Save"}
          </button>
        </div>
        <ActionNote state={stateOf("token")} onDismiss={() => dismiss("token")} />
        <code className="library-path">{state.hfHome}/token</code>
      </section>
    </>
  );
}

/** The backend's last published choice, which is its widest. Never a local list. */
function lastChoice(model: ModelStatus): number {
  const choices = model.quantization.prequantize_choices;
  return choices[choices.length - 1] ?? 8;
}

/**
 * A real switch.
 *
 * `role="switch"` with `aria-checked` rather than a styled checkbox or, as
 * before, an ON/OFF pill that looked interactive and was not: the state has to be
 * in the accessibility tree, not only in the pixels.
 */
function Switch({
  checked,
  label,
  disabled,
  onChange,
}: {
  checked: boolean;
  label: string;
  disabled?: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="switch"
      disabled={disabled}
      onClick={() => onChange(!checked)}
    />
  );
}

type RowProps = {
  model: ModelStatus;
  caps: ModelCapabilities | undefined;
  isDefault: boolean;
  tokenPresent: boolean;
  anyBusy: boolean;
  jobActive: boolean;
  busy: (key: string) => boolean;
  stateOf: (key: string) => Parameters<typeof ActionNote>[0]["state"];
  dismiss: (key: string) => void;
  /** This model's entry in the configuration, or an empty one. */
  override: Record<string, any>;
  /** The running server exposes this model. `null` when there is no server to ask. */
  servedByServer: boolean | null;
  bits: number;
  onBits: (value: number) => void;
  selectedComponents: string[];
  onComponents: (next: (previous: string[]) => string[]) => void;
  onDownload: () => void;
  onConvert: (bits: number) => void;
  onActivate: (bits: number | null) => void;
  onForget: () => void;
  onPatch: (patch: Record<string, unknown>, key: string, success?: string) => void;
  /** This built-in reads from a local `model_path` override. */
  located: boolean;
  onLocate: () => void;
  onResetLocation: () => void;
};

/**
 * One model: what it is, whether its weights are here, what it can be converted
 * to, which representation generation will use, and how it is configured.
 *
 * Everything model-specific lives here now. It used to be split — availability
 * and conversion in this view, `enabled`/quantize/steps/guidance in a second
 * table under Configuration — which meant turning a model on and downloading it
 * were two screens apart, and the same model appeared twice under two different
 * names for two different halves of itself.
 *
 * Every branch below reads a backend field. Nothing re-derives whether a download
 * is offered (`can_download`), which bit depths exist (`prequantize_choices`),
 * whether conversion is possible (`supports_prequantize`) or which variant is
 * live (`active_variant`).
 */
function ModelRow({
  model,
  caps,
  isDefault,
  tokenPresent,
  anyBusy,
  jobActive,
  busy,
  stateOf,
  dismiss,
  override,
  servedByServer,
  bits,
  onBits,
  selectedComponents,
  onComponents,
  onDownload,
  onConvert,
  onActivate,
  onForget,
  onPatch,
  located,
  onLocate,
  onResetLocation,
}: RowProps) {
  const [open, setOpen] = useState(false);
  const action = ACTION[model.availability];
  // A gated repository with no token would 401 several minutes in: better to say
  // so before starting. Gating is a HuggingFace concept, so an imported local
  // model — which has no repository — can never be blocked by it.
  const blocked = model.can_download && model.gated && !tokenPresent;
  const quant = model.quantization;
  const memoryBounded = quant.prequantize_strategy === "qds_memory_bounded";
  const imported = model.provenance === "imported_local";
  const enabled = override.enabled !== false;

  // Two backend facts disagreeing, not a flag of our own: the configuration says
  // this model should be exposed, and `/v1/capabilities` — which lists what the
  // *running* server actually loaded its registry with — says otherwise. That is
  // exactly what a restart would fix, and it is the only thing said about it.
  const restartRequired = servedByServer !== null && servedByServer !== enabled;

  const settingsId = `settings-${model.key}`;

  return (
    <li className={`model${enabled ? "" : " off"}`}>
      <div className="model-head">
        <span className="model-title">{model.display_name}</span>
        <span className={`pill ${action.tone}`} title={model.detail ?? undefined}>
          {model.availability === "present" && model.size_gb ? `${model.size_gb} GB` : action.badge}
        </span>
        {isDefault && <span className="pill pill-accent">default</span>}
        {imported && <span className="pill">imported local</span>}
        {model.gated && <span className="pill pill-warn">gated</span>}
        {/* Beside the identity rather than after the conversion button: it says
            what this model *is* — weights that already carry their precision, so
            a runtime quantize setting would be ignored. */}
        {!quant.supports_quantization && quant.note && (
          <span className="pill" title={quant.note}>
            fixed precision
          </span>
        )}
        {restartRequired && (
          <span
            className="pill pill-warn"
            title="Saved. The running generation server still has its previous model set."
          >
            restart required
          </span>
        )}

        <span className="model-enable">
          <span className="label" id={`enable-label-${model.key}`}>
            {enabled ? "Enabled" : "Disabled"}
          </span>
          <Switch
            checked={enabled}
            label={`Enable ${model.display_name}`}
            disabled={anyBusy}
            onChange={(next) => onPatch({ enabled: next }, `enable:${model.key}`)}
          />
        </span>
      </div>

      {/* For a built-in that is its repository; for an imported model, where it
          actually lives. Both are the source identity. */}
      <p className="model-meta">
        <code>{model.repo}</code>
        {model.family && (
          <>
            <span className="sep">·</span>
            <span>{model.family}</span>
          </>
        )}
        {imported ? (
          <>
            {/* What to put in `{"model": ...}`. The internal id is deliberately
                not shown as the thing to send. */}
            <span className="sep">·</span>
            <span>
              API name <code>{model.api_name}</code>
            </span>
            {model.base_profile_key && (
              <>
                <span className="sep">·</span>
                <span>defaults from {model.base_profile_key}</span>
              </>
            )}
          </>
        ) : model.license ? (
          <>
            <span className="sep">·</span>
            <span>{model.license}</span>
          </>
        ) : null}
        {caps && (
          <>
            <span className="sep">·</span>
            <span>
              {caps.default_steps} steps
              {caps.supports_guidance && caps.default_guidance !== null
                ? ` · guidance ${caps.default_guidance}`
                : ""}
              {caps.quantize ? ` · ${caps.quantize}-bit` : ""}
            </span>
          </>
        )}
      </p>

      {/* The server's own words. Paraphrasing loses the part that says what to do
          about it — an unplugged volume in particular. */}
      {model.detail && model.availability !== "present" && (
        <p className="library-detail">{model.detail}</p>
      )}

      <div className="model-actions">
        {/* A local artifact has nothing to fetch: its path either holds a
            completed artifact or it does not. */}
        {action.label && model.can_download && (
          <span className="model-group">
            <button
              className="small"
              onClick={onDownload}
              disabled={anyBusy || jobActive}
              title={
                blocked
                  ? "This repository is gated: save a HuggingFace token below first, or the download will fail with a 401."
                  : jobActive
                    ? "Another long operation is already running."
                    : (model.detail ?? undefined)
              }
            >
              {busy(`download:${model.key}`)
                ? "Starting…"
                : blocked
                  ? `${action.label} ⚠`
                  : action.label}
            </button>
          </span>
        )}

        {!imported && action.label && model.can_download && (
          <span className="model-group">
            <button className="small" onClick={onLocate} disabled={anyBusy || jobActive}>
              {busy(`locate:${model.key}`) ? "Checking…" : "Locate…"}
            </button>
          </span>
        )}

        {!imported && located && (
          <span className="model-group">
            <button
              className="small"
              onClick={onResetLocation}
              disabled={anyBusy}
              title="Forgets the folder and goes back to the catalogue repository. No files are deleted."
            >
              {busy(`locate:${model.key}`) ? "Resetting…" : "Reset location"}
            </button>
          </span>
        )}

        {quant.supports_prequantize && (
          <span className="model-group">
            <select
              id={`bits-${model.key}`}
              aria-label={`Bit depth for ${model.key}`}
              value={String(bits)}
              onChange={(event) => onBits(Number(event.target.value))}
            >
              {quant.prequantize_choices.map((choice) => (
                <option key={choice} value={String(choice)}>
                  {choice} bits
                </option>
              ))}
            </select>
            <button
              className="small primary"
              onClick={() => onConvert(bits)}
              disabled={anyBusy || jobActive || (memoryBounded && selectedComponents.length === 0)}
              title={jobActive ? "Another long operation is already running." : undefined}
            >
              {busy(`convert:${model.key}`) ? "Starting…" : "Pre-quantize"}
            </button>
          </span>
        )}

        {/* Source, saved variants and the active representation are three
            different facts, so the active one is marked rather than merely
            selected. */}
        {model.variants.length > 0 && (
          <span className="model-group">
            <span className="group-label">Saved variants</span>
            {model.variants.map((variant) => (
              <button
                key={variant.bits}
                className="small variant"
                aria-pressed={model.active_variant === variant.bits}
                onClick={() => onActivate(variant.bits)}
                disabled={anyBusy || model.active_variant === variant.bits}
                title={variant.path}
              >
                {variant.bits}-bit
                {model.active_variant === variant.bits ? " · active" : ""}
              </button>
            ))}
            {model.active_variant !== null && (
              <button className="small" onClick={() => onActivate(null)} disabled={anyBusy}>
                Use original
              </button>
            )}
          </span>
        )}

        {imported && (
          <span className="model-group">
            <button
              className="small"
              onClick={onForget}
              disabled={anyBusy}
              title="Removes the registration only. The model files are not deleted."
            >
              {busy(`forget:${model.key}`) ? "Removing…" : "Forget"}
            </button>
          </span>
        )}

        {/* Kept in this row rather than on another screen: they are settings for
            this model, and the core state above stays visible while they are
            open. */}
        <span className="model-group" style={{ marginLeft: "auto" }}>
          <button
            className="small"
            aria-expanded={open}
            aria-controls={settingsId}
            onClick={() => setOpen((previous) => !previous)}
          >
            Model settings {open ? "▴" : "▾"}
          </button>
        </span>
      </div>

      {open && (
        <div className="model-advanced" id={settingsId}>
          <ModelSettings
            model={model}
            caps={caps}
            override={override}
            busy={busy(`settings:${model.key}`)}
            disabled={anyBusy}
            onApply={(patch) =>
              onPatch(patch, `settings:${model.key}`, "Saved. It applies when the server restarts.")
            }
          />
        </div>
      )}

      {memoryBounded && (
        <fieldset className="settings-group">
          <legend>Components to convert</legend>
          <div className="row">
            {COMPONENTS.map((component) => (
              <label className="check" key={component.id}>
                <input
                  type="checkbox"
                  checked={selectedComponents.includes(component.id)}
                  onChange={(event) =>
                    onComponents((previous) =>
                      event.target.checked
                        ? COMPONENTS.filter(
                            (item) => previous.includes(item.id) || item.id === component.id,
                          ).map((item) => item.id)
                        : previous.filter((id) => id !== component.id),
                    )
                  }
                />
                <span title={component.detail}>{component.label}</span>
              </label>
            ))}
          </div>
          <p className="setting-help">
            Converted one component at a time and quantized block by block: these bf16 weights are
            far larger than unified memory.
          </p>
        </fieldset>
      )}

      {["download", "locate", "convert", "activate", "forget", "enable", "settings"].map((kind) => (
        <ActionNote
          key={kind}
          state={stateOf(`${kind}:${model.key}`)}
          onDismiss={() => dismiss(`${kind}:${model.key}`)}
        />
      ))}
    </li>
  );
}

/**
 * The per-model overrides, edited as a draft and applied in one write.
 *
 * A draft rather than a field-by-field write because these are typed values: a
 * write per keystroke would rewrite the configuration file five times while
 * someone types "1024" and hand the server a `1` to validate on the way.
 *
 * Which controls are offered, and which are inert, is the backend's declaration —
 * `quantize_choices`, `supports_guidance`, `supports_edit`. Nothing here decides
 * what a model can do.
 */
function ModelSettings({
  model,
  caps,
  override,
  busy,
  disabled,
  onApply,
}: {
  model: ModelStatus;
  caps: ModelCapabilities | undefined;
  override: Record<string, any>;
  busy: boolean;
  disabled: boolean;
  onApply: (patch: Record<string, unknown>) => void;
}) {
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const value = (key: string) => (key in draft ? draft[key] : override[key]);
  const set = (key: string, next: unknown) => setDraft((previous) => ({ ...previous, [key]: next }));
  const dirty = Object.keys(draft).length > 0;

  const quant = model.quantization;
  const configured = value("quantize");
  const quantizeValue = configured === null || configured === undefined ? "" : String(configured);
  // A stored value the backend no longer publishes is kept visible and marked,
  // rather than silently coerced.
  const stale =
    typeof configured === "number" &&
    configured !== 0 &&
    !quant.quantize_choices.includes(configured)
      ? configured
      : null;

  const numeric = (key: string) => ({
    type: "number" as const,
    value: (value(key) ?? "") as string | number,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      set(key, event.target.value === "" ? null : Number(event.target.value)),
  });

  return (
    <>
      <div className="setting-pair">
        <div className="setting">
          <label className="setting-label" htmlFor={`quantize-${model.key}`}>
            Quantization
          </label>
          {!quant.supports_quantization ? (
            <p className="setting-help" style={{ marginTop: 0 }}>
              {quant.note ?? "Fixed for this model."}
            </p>
          ) : (
            <>
              <select
                id={`quantize-${model.key}`}
                aria-label={`Quantization for ${model.key}`}
                value={quantizeValue}
                onChange={(event) =>
                  set("quantize", event.target.value === "" ? null : Number(event.target.value))
                }
              >
                <option value="">default</option>
                <option value="0">none (bf16)</option>
                {quant.quantize_choices.map((choice) => (
                  <option key={choice} value={String(choice)}>
                    {choice} bits
                  </option>
                ))}
                {stale !== null && (
                  <option value={String(stale)}>{stale} bits (invalid)</option>
                )}
              </select>
              {stale !== null && (
                <p className="setting-error">not supported by this model</p>
              )}
            </>
          )}
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor={`steps-${model.key}`}>
            Steps
          </label>
          <input
            id={`steps-${model.key}`}
            aria-label={`Steps for ${model.key}`}
            min={1}
            placeholder={caps ? String(caps.default_steps) : "model default"}
            {...numeric("default_steps")}
          />
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor={`guidance-${model.key}`}>
            Guidance
          </label>
          <input
            id={`guidance-${model.key}`}
            aria-label={`Guidance for ${model.key}`}
            min={0}
            step={0.5}
            // Distilled model: the server rejects any value.
            disabled={caps ? !caps.supports_guidance : false}
            placeholder={
              caps?.supports_guidance === false
                ? `fixed ${caps.default_guidance ?? 0}`
                : caps
                  ? String(caps.default_guidance ?? "")
                  : "model default"
            }
            {...numeric("default_guidance")}
          />
          {caps?.supports_guidance === false && (
            <p className="setting-help">This model ignores guidance.</p>
          )}
        </div>

        <div className="setting">
          <span className="setting-label">Image editing</span>
          <label className="check">
            <input
              type="checkbox"
              aria-label={`Editing for ${model.key}`}
              disabled={caps ? !caps.supports_edit && override.enable_edit !== true : false}
              checked={value("enable_edit") === true}
              onChange={(event) => set("enable_edit", event.target.checked)}
            />
            <span>Expose the edits endpoint</span>
          </label>
          {caps?.supports_edit === false && (
            <p className="setting-help">Not supported by this model.</p>
          )}
        </div>
      </div>

      <div className="actions" style={{ marginTop: 12 }}>
        <button
          className="small primary"
          onClick={() => onApply(draft)}
          disabled={!dirty || busy || disabled}
        >
          {busy ? "Saving…" : "Apply"}
        </button>
        <button className="small" onClick={() => setDraft({})} disabled={!dirty || busy}>
          Reset
        </button>
        <span className="setting-help" style={{ margin: 0 }}>
          Applies when the generation server restarts.
        </span>
      </div>
    </>
  );
}

function ImportConfirmation({
  verdict,
  profile,
  name,
  apiName,
  capabilities,
  busy,
  onProfile,
  onName,
  onApiName,
  onConfirm,
  onCancel,
}: {
  verdict: ImportVerdict;
  profile: string;
  name: string;
  apiName: string;
  capabilities: Capabilities | null;
  busy: boolean;
  onProfile: (value: string) => void;
  onName: (value: string) => void;
  onApiName: (value: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <fieldset className="settings-group">
      <legend>Confirm import</legend>

      <div className="setting">
        <span className="setting-label">Folder</span>
        <code className="library-path">{verdict.path}</code>
      </div>

      <div className="setting">
        <span className="setting-label">Detected family</span>
        <div className="row">
          <span className="pill pill-accent">{verdict.family}</span>
          <span className="setting-help" style={{ margin: 0 }}>
            from <code>_class_name</code>: <code>{verdict.class_name}</code>
          </span>
        </div>
      </div>

      <div className="setting-pair">
        <div className="setting">
          <label className="setting-label" htmlFor="import-profile">
            Base profile
          </label>
          <select
            id="import-profile"
            aria-label="Base profile"
            value={profile}
            onChange={(event) => onProfile(event.target.value)}
          >
            <option value="">choose…</option>
            {verdict.profiles.map((candidate) => (
              <option key={candidate} value={candidate}>
                {candidate}
                {capabilities?.models[candidate]
                  ? ` — ${capabilities.models[candidate]!.default_steps} steps`
                  : ""}
              </option>
            ))}
          </select>
          <p className="setting-help">
            Supplies this model's generation defaults — steps, guidance, scheduler.
          </p>
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="import-name">
            Display name
          </label>
          <input
            id="import-name"
            type="text"
            value={name}
            onChange={(event) => onName(event.target.value)}
          />
          <p className="setting-help">How it appears in this list.</p>
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="import-api-name">
            API name
          </label>
          <input
            id="import-api-name"
            type="text"
            value={apiName}
            spellCheck={false}
            onChange={(event) => onApiName(event.target.value)}
          />
          {/* Three identities, and this is the machine-facing one. The internal
              id stays opaque and durable; the display name is for people. */}
          <p className="setting-help">
            What API requests send as <code>"model"</code>. Lowercase letters, digits,{" "}
            <code>.</code>, <code>_</code> or <code>-</code>, and unique across every model.
          </p>
        </div>
      </div>

      <div className="actions" style={{ marginTop: 14 }}>
        <button className="primary" onClick={onConfirm} disabled={!profile || !apiName || busy}>
          Register
        </button>
        <button onClick={onCancel}>Cancel</button>
        {!profile && (
          <span className="setting-help" style={{ margin: 0 }}>
            Choose the profile whose generation defaults this model should use.
          </span>
        )}
      </div>
    </fieldset>
  );
}

/** The same conservative slug the backend derives, for the live default. */
function slugFor(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "")
    .slice(0, 64);
}

/**
 * Confirming that a directory may be bound to a built-in catalogue entry.
 *
 * The distinction this surface exists to make: a cached repository *proves* which
 * repository it is, because huggingface_hub encodes it in the directory name. A
 * folder of compatible weights proves nothing of the sort, and saying "this is
 * FLUX.2-klein" about it would be QDS's assertion, not a fact. So the unproven
 * case is stated before the binding, not hidden behind a success message.
 */
function LocateConfirmation({
  verdict,
  busy,
  onConfirm,
  onCancel,
}: {
  verdict: LocateVerdict;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <fieldset className="settings-group">
      <legend>Use this folder for {verdict.model}</legend>

      <div className="setting">
        <span className="setting-label">Folder</span>
        <code className="library-path">{verdict.path}</code>
      </div>

      <div className="setting">
        <span className="setting-label">Detected</span>
        <div className="row">
          <span className="pill pill-accent">{verdict.family}</span>
          <span className="setting-help" style={{ margin: 0 }}>
            from <code>_class_name</code>: <code>{verdict.class_name}</code>
          </span>
        </div>
      </div>

      {verdict.repo_verified ? (
        <p className="setting-help">
          This folder is a Hugging Face cache entry for <code>{verdict.detected_repo}</code> — the
          same repository the catalogue names, so its identity is confirmed.
        </p>
      ) : (
        <p className="caution">
          QDS can confirm this folder holds a compatible <strong>{verdict.family}</strong> model,
          but not that it is the exact repository the catalogue names
          {verdict.detected_repo ? (
            <>
              {" "}
              — the folder identifies itself as <code>{verdict.detected_repo}</code>
            </>
          ) : (
            " — it carries no Hugging Face cache metadata to check against"
          )}
          . Generation defaults will come from the catalogue entry regardless.
        </p>
      )}

      <div className="actions" style={{ marginTop: 14 }}>
        <button className="primary" onClick={onConfirm} disabled={busy}>
          {busy ? "Saving…" : "Use this folder"}
        </button>
        <button onClick={onCancel}>Cancel</button>
        <span className="setting-help" style={{ margin: 0 }}>
          Nothing is copied or moved.
        </span>
      </div>
    </fieldset>
  );
}
