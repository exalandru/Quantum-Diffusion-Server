import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api";
import { PathPrompt } from "../PathPrompt";
import { messageOf } from "../api";
import { ActionNote, useActions } from "../actions";
import { describeJob, describeOutcome, type JobView } from "../job";
import { Modal } from "../modal";
import { ImportConfirmation } from "./models/ImportConfirmation";
import { LocateConfirmation } from "./models/LocateConfirmation";
import { QuantizationDialog } from "./models/QuantizationDialog";
import { SizeBadge } from "./models/SizeBadge";
import { Switch } from "./models/Switch";
import { ACTION, OPEN_TAB, CATALOGUE_TABS, lastChoice, slugFor } from "./models/shared";
import type { CatalogueTab, RowProps } from "./models/shared";
import type {
  Capabilities,
  ConfigWarning,
  ImportVerdict,
  JobStatus,
  ModelCapabilities,
  LocateVerdict,
  ModelStatus,
  Overview,
} from "../types";

export function Models({
  state,
  config,
  jobs,
  onConfigChanged,
}: {
  state: Overview;
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
  /** Runtime invariants this configuration breaks. Not errors of this view. */
  const [configWarnings, setConfigWarnings] = useState<ConfigWarning[]>([]);
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
  /** Which half of the catalogue is on screen. Open first: see `CATALOGUE_TABS`. */
  const [tab, setTab] = useState<CatalogueTab["id"]>("open");
  /** Once the API name is typed in, it stops following the display name. */
  const apiNameEdited = useRef(false);
  /** Which path is being asked for, if any. Replaces the native folder chooser. */
  const [prompt, setPrompt] = useState<
    { kind: "import" } | { kind: "locate"; model: ModelStatus } | null
  >(null);

  useEffect(() => {
    void api
      .capabilities()
      .then((value) => {
        setCapabilities(value);
        setCapabilitiesNote(null);
      })
      .catch((cause) => {
        setCapabilities(null);
        setCapabilitiesNote(messageOf(cause));
      });
  }, []);

  // The catalogue is computed by the server without importing mflux: it lists the
  // disabled models too, and it works with the server stopped — which is when you
  // most want to download weights.
  const reloadModels = useCallback(async () => {
    try {
      const status = await api.modelsStatus();
      setModels(status.models);
      // Reported beside the rows rather than instead of them: these are reasons
      // the *generation server* would refuse to start, and the controls that fix
      // them are on this screen and in Configuration.
      setConfigWarnings(status.warnings ?? []);
      setCatalogueError(null);
    } catch (cause) {
      setCatalogueError(messageOf(cause));
    }
  }, []);

  useEffect(() => {
    void reloadModels();
  }, [reloadModels]);

  // Any finished operation changes what is on disk, so the catalogue is stale.
  //
  // This used to read `settled.kind === "fetch"`, which is the bug: a conversion
  // that completed — writing a component, finishing a variant, and having its
  // selection recorded by the supervisor — left every one of those facts
  // unasked-for. The row kept its old variants, the dialog kept its old
  // component states, and the only way to see the truth was to remount the
  // panel by switching tabs, or to relaunch.
  //
  // Both refreshes, because a conversion changes two different things: the
  // artifact on disk, which only `models_status` knows about, and
  // `prequantized_variant` in the configuration, which the supervisor writes on
  // completion and which `App` owns. Nothing is cached here and nothing is
  // inferred from the job — both calls re-read the backend.
  useEffect(
    () =>
      jobs.onSettled((settled) => {
        if (settled.state !== "completed") return;
        void reloadModels();
        void onConfigChanged();
      }),
    [jobs, reloadModels, onConfigChanged],
  );

  // The action covers only *starting* the job — that is what can be refused
  // (single-flight) or fail to spawn. How it ends is the server's to report, and
  // survives this panel unmounting.
  async function download(key: string) {
    if (await run(`download:${key}`, () => api.modelFetch(key))) {
      setDismissedAt(null);
      await jobs.refresh();
    }
  }

  async function importFrom(path: string) {
    setPrompt(null);
    await run("import", async () => {
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
  async function locateFrom(model: ModelStatus, path: string) {
    setPrompt(null);
    await run(`locate:${model.key}`, async () => {
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

  async function convert(model: ModelStatus, bits: number, components: string[]) {
    // Exactly what the dialog selected, and the backend orders and validates it:
    // every supported family converts component by component now, so a subset is
    // a legitimate run rather than a claim the whole model was written.
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

  const job = jobs.job;

  /**
   * What a finished operation achieved, in the backend's terms.
   *
   * The component labels come from the model the job names — which is the same
   * published list the dialog renders — so a partial run can say "Text encoder
   * converted" without this file knowing what a text encoder is. And when the
   * server is running, a completed variant says plainly that the live process is
   * not using it yet: the configuration has changed, the running registry has
   * not, and only a restart reconciles them.
   */
  function outcomeOf(finished: JobStatus): string {
    const key = (finished.fields as { model?: string } | null)?.model;
    const specs =
      (models ?? []).find((row) => row.key === key)?.quantization.prequantize_components ?? [];
    const outcome = describeOutcome(
      finished,
      (component) => specs.find((spec) => spec.key === component)?.label ?? component,
    );
    if (!outcome) return `${finished.target ?? "Operation"} finished.`;
    if (finished.event === "prequantize_done") {
      return `${outcome} Restart the server to generate with it.`;
    }
    return outcome;
  }
  // The server settled a terminal outcome the user has not dismissed. Shown through the
  // same note as an action result, but sourced from the server — polling re-reads it
  // rather than clearing it.
  const terminal =
    job && !jobs.active && job.state !== "idle" && job.finishedAtMs !== dismissedAt ? job : null;

  const builtIn = (models ?? []).filter((model) => model.provenance === "built_in");
  const imported = (models ?? []).filter((model) => model.provenance === "imported_local");
  // Three groups, and the split is a backend fact in every case: `provenance`
  // separates what the user registered from what QDS ships, and `gated`
  // separates the repositories that need granted access from the ones that do
  // not. Filtering preserves the catalogue's own order inside each group, which
  // is the backend's and not ours to reorder.
  const gated = builtIn.filter((model) => model.gated);
  const ungated = builtIn.filter((model) => !model.gated);
  // `?? OPEN_TAB` rather than an index: the state only ever holds one of the
  // table's own ids, and a named default is total where `[0]` is not.
  const visibleTab = CATALOGUE_TABS.find(({ id }) => id === tab) ?? OPEN_TAB;

  const overrideOf = (key: string): Record<string, any> => {
    const models = (config as { models?: Record<string, any> } | null)?.models;
    return models?.[key] ?? {};
  };

  // What the *running* server exposes, which is not what the configuration says
  // until it has been restarted. `null` when there is no answer to compare with.
  const servedBy = (key: string): boolean | null =>
    capabilities ? key in capabilities.models : null;

  const rowProps = (model: ModelStatus): RowProps => ({
    model,
    caps: capabilities?.models[model.key],
    isDefault: model.key === capabilities?.default_model,
    tokenPresent: state.hfTokenPresent,
    anyBusy,
    job,
    jobActive: jobs.active,
    cancelBusy: busy("cancel"),
    onCancelJob: () => void run("cancel", async () => void (await api.jobCancel())),
    busy,
    stateOf,
    dismiss,
    override: overrideOf(model.key),
    servedByServer: servedBy(model.key),
    bits: bitsFor[model.key] ?? lastChoice(model),
    onBits: (value: number) => setBitsFor((previous) => ({ ...previous, [model.key]: value })),
    onDownload: () => void download(model.key),
    onConvert: (value: number, componentKeys: string[]) =>
      void convert(model, value, componentKeys),
    onActivate: (value: number | null) => void activate(model, value),
    onForget: () => void forget(model),
    located: typeof overrideOf(model.key).model_path === "string",
    onLocate: () => setPrompt({ kind: "locate", model }),
    onResetLocation: () => void resetLocation(model),
    onPatch: (patch: Record<string, unknown>, key: string, success?: string) =>
      void patchModel(model, patch, key, success),
  });

  return (
    <>
      {/* ── The server-owned operation ───────────────────────────────────── */}
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
                  ? { status: "ok", message: outcomeOf(terminal) }
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

        {/* What the generation server would refuse to start on. The catalogue
            renders regardless — this used to be a traceback where the list
            should have been, which took away the switches that repair it. */}
        {configWarnings.map((warning) => (
          <div className="notice notice-warn" role="status" key={warning.code}>
            <strong>Configuration problem.</strong> {warning.message}
          </div>
        ))}

        {/* The token itself is a global source setting and lives in
            Configuration. What belongs here is only the consequence: those
            particular rows cannot be downloaded until it exists. It names the
            Gated tab rather than "these repositories", which stopped being true
            when the halves became tabs — the rows it is about are usually not the
            ones on screen. */}
        {gated.length > 0 && !state.hfTokenPresent && (
          <p className="caution">
            <span className="pill pill-warn">no token</span> The {gated.length} repositories in the{" "}
            <strong>Gated</strong> tab will refuse a download with a 401. Add a Hugging Face token
            under <strong>Configuration → Hugging Face</strong>, then grant your account access on
            each model's card.
          </p>
        )}

        {prompt && (
          <PathPrompt
            title={prompt.kind === "import" ? "Import a local model" : "Locate existing weights"}
            hint={
              prompt.kind === "import"
                ? "The directory holding the model's files. It is registered where it is - nothing is copied or moved."
                : `Where ${prompt.model.key}'s weights already live on this machine. The directory is checked against that entry before anything is bound.`
            }
            placeholder={prompt.kind === "locate" ? state.effectiveHfHome : undefined}
            onCancel={() => setPrompt(null)}
            onSubmit={(path) =>
              prompt.kind === "import"
                ? void importFrom(path)
                : void locateFrom(prompt.model, path)
            }
          />
        )}

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
          <>
            {/* One list at a time, and Open is the one that opens: those can be
                installed right now, with no account and nothing to request. The
                two groups used to be stacked, which meant scrolling past five
                repositories you may have no access to on the way to the ones you
                can use. Each list keeps the catalogue's own order — this changes
                what is read first, not what the backend publishes.

                The shell's own tab vocabulary (`views`/`view-tab`, `role=tab`,
                `aria-selected`), so there is one way tabs look and behave in this
                app rather than a second one invented here. */}
            <nav className="views catalogue-tabs" role="tablist" aria-label="Catalogue">
              {CATALOGUE_TABS.map(({ id, label, models: pick }) => (
                <button
                  key={id}
                  role="tab"
                  aria-selected={tab === id}
                  // A stable accessible name: the count changes as models are
                  // imported or forgotten, and a control that renames itself is
                  // one a screen reader cannot address.
                  aria-label={label}
                  className="view-tab"
                  onClick={() => setTab(id)}
                >
                  {label}
                  <span className="count" aria-hidden="true">
                    {pick({ gated, ungated }).length}
                  </span>
                </button>
              ))}
            </nav>
            <CatalogueGroup
              heading={visibleTab.heading}
              note={visibleTab.note}
              models={visibleTab.models({ gated, ungated })}
              rowProps={rowProps}
            />
          </>
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
          <button onClick={() => setPrompt({ kind: "import" })} disabled={busy("import")}>
            {busy("import") ? "Working…" : "Import Local Model…"}
          </button>
        </div>
        <p className="note">
          Register a model already on this machine. Nothing is copied - QDS records where it is, and
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
    </>
  );
}

/**
 * One catalogue section.
 *
 * The grouping is what makes the list scannable, and it is drawn from the
 * backend's own `gated` flag rather than from a list kept here: a repository
 * that stops being gated moves group on the next catalogue read, with nothing
 * to update. An empty group renders nothing at all rather than a heading over
 * a hole.
 */
function CatalogueGroup({
  heading,
  note,
  models,
  rowProps,
}: {
  heading: string;
  note: string;
  models: ModelStatus[];
  rowProps: (model: ModelStatus) => RowProps;
}) {
  // No early return for an empty group any more: it is now the *selected* tab's
  // panel, and a tab that renders nothing at all reads as a broken view rather
  // than as an empty one.
  return (
    <section className="catalogue-group">
      <h3 className="catalogue-heading">{heading}</h3>
      <p className="note" style={{ marginTop: 0 }}>
        {note}
      </p>
      {models.length === 0 ? (
        <p className="empty">Nothing in this half of the catalogue.</p>
      ) : (
        <ul className="models" aria-label={heading}>
          {models.map((model) => (
            <ModelRow key={model.key} {...rowProps(model)} />
          ))}
        </ul>
      )}
    </section>
  );
}

/**
 * One model, read top to bottom: what it is, what state its weights are in,
 * where it comes from, and what can be done to it.
 *
 * Four bands, in that order, because they answer four different questions and
 * used to answer them on one baseline: the name sat in a row of pills, so
 * scanning ten models for the one you wanted meant reading past forty badges.
 * The name is now a heading with nothing beside it but its own switch; the state
 * is a band under it; the source is a band under that; the actions are last.
 *
 * The technical controls are behind two dialogs rather than in the row. A bit
 * depth selector and a Pre-quantize button in a catalogue row invite the reading
 * that they are part of installing it, which is how someone starts a two-hour
 * conversion meaning to download a model.
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
  job,
  jobActive,
  cancelBusy,
  onCancelJob,
  busy,
  stateOf,
  dismiss,
  override,
  servedByServer,
  bits,
  onBits,
  onDownload,
  onConvert,
  onActivate,
  onForget,
  onPatch,
  located,
  onLocate,
  onResetLocation,
}: RowProps) {
  const [dialog, setDialog] = useState<"quantization" | "settings" | null>(null);
  const action = ACTION[model.availability];
  // A gated repository with no token would 401 several minutes in: better to say
  // so before starting. Gating is a HuggingFace concept, so an imported local
  // model — which has no repository — can never be blocked by it.
  const blocked = model.can_download && model.gated && !tokenPresent;
  const quant = model.quantization;
  const imported = model.provenance === "imported_local";
  const enabled = override.enabled !== false;

  // Two backend facts disagreeing, not a flag of our own: the configuration says
  // this model should be exposed — and which representation it should use — while
  // `/v1/capabilities` reports what the *running* server actually loaded its
  // registry with. That is exactly what a restart would fix, and it is the only
  // thing said about it.
  //
  // The variant half matters most right after a conversion: the supervisor
  // selects the new copy the moment it validates, and a running process cannot
  // have adopted a choice made after it started. Without this comparison the row
  // would show the new variant as active while generation kept using the old
  // representation.
  // A server that does not publish `active_variant` at all — an older
  // environment than this app — has told us nothing, and nothing is not a
  // disagreement. Present-and-null is a real answer ("the source") and is
  // compared; absent is left alone rather than turned into a permanent and false
  // "restart required".
  const liveVariant =
    caps !== undefined && "active_variant" in caps ? caps.active_variant : undefined;
  const restartRequired =
    servedByServer !== null &&
    (servedByServer !== enabled ||
      (liveVariant !== undefined && liveVariant !== model.active_variant));

  // Nothing to say about precision for a model that can neither be quantized on
  // load nor pre-quantized, and that has no saved copy to activate.
  const hasQuantization =
    quant.supports_quantization || quant.supports_prequantize || model.variants.length > 0;

  // The one fact from inside the dialog worth carrying in the row: which
  // representation generation will actually use.
  const variantSummary =
    model.active_variant !== null
      ? `using the ${model.active_variant}-bit copy`
      : model.variants.length > 0
        ? `${model.variants.length} saved ${model.variants.length === 1 ? "copy" : "copies"}`
        : null;

  return (
    <li className={`model${enabled ? "" : " off"}`}>
      {/* ── Identity ─────────────────────────────────────────────────────── */}
      <div className="model-head">
        <h3 className="model-name">{model.display_name}</h3>
        <span className="model-enable">
          <span className="label">{enabled ? "Enabled" : "Disabled"}</span>
          <Switch
            checked={enabled}
            label={`Enable ${model.display_name}`}
            disabled={anyBusy}
            onChange={(next) => onPatch({ enabled: next }, `enable:${model.key}`)}
          />
        </span>
      </div>

      {/* ── State ────────────────────────────────────────────────────────── */}
      <div className="model-badges">
        <span className={`pill ${action.tone}`} title={model.detail ?? undefined}>
          {action.badge}
        </span>
        {/* The measured size, separate from the availability it used to replace:
            "12.5 GB" answered "how big", never "is it here". What it reports is
            the *active* representation — a model set to its 4-bit copy loads
            5.9 GB, whatever its 20.5 GB source still occupies — and the rest of
            the accounting is a disclosure away rather than absent. */}
        <SizeBadge disk={model.disk} activeVariant={model.active_variant} />
        {model.availability !== "present" && model.size_gb > 0 && (
          <span className="pill" title="Size of the repository on Hugging Face, not on this disk.">
            {model.size_gb} GB to download
          </span>
        )}
        {isDefault && <span className="pill pill-accent">default</span>}
        {model.gated && <span className="pill pill-warn">gated</span>}
        {imported && <span className="pill">imported local</span>}
        {/* What this model *is* — weights that already carry their precision, so
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
      </div>

      {/* ── Source ───────────────────────────────────────────────────────── */}
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

      {/* ── Actions ──────────────────────────────────────────────────────── */}
      <div className="model-actions">
        {/* A local artifact has nothing to fetch: its path either holds a
            completed artifact or it does not. */}
        {action.label && model.can_download && (
          <button
            className="small"
            onClick={onDownload}
            disabled={anyBusy || jobActive}
            title={
              blocked
                ? "This repository is gated: add a Hugging Face token in Configuration first, or the download will fail with a 401."
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
        )}

        {!imported && action.label && model.can_download && (
          <button className="small" onClick={onLocate} disabled={anyBusy || jobActive}>
            {busy(`locate:${model.key}`) ? "Checking…" : "Locate…"}
          </button>
        )}

        {!imported && located && (
          <button
            className="small"
            onClick={onResetLocation}
            disabled={anyBusy}
            title="Forgets the folder and goes back to the catalogue repository. No files are deleted."
          >
            {busy(`locate:${model.key}`) ? "Resetting…" : "Reset location"}
          </button>
        )}

        {hasQuantization && (
          <button className="small" onClick={() => setDialog("quantization")}>
            Quantization…
          </button>
        )}

        <button className="small" onClick={() => setDialog("settings")}>
          Model settings…
        </button>

        {imported && (
          <button
            className="small"
            onClick={onForget}
            disabled={anyBusy}
            title="Removes the registration only. The model files are not deleted."
          >
            {busy(`forget:${model.key}`) ? "Removing…" : "Forget"}
          </button>
        )}

        {variantSummary && <span className="model-summary">{variantSummary}</span>}
      </div>

      {/* Only the outcomes of the actions this row still performs. A conversion's
          or a saved copy's belongs in the dialog that started it. */}
      {["download", "locate", "forget", "enable"].map((kind) => (
        <ActionNote
          key={kind}
          state={stateOf(`${kind}:${model.key}`)}
          onDismiss={() => dismiss(`${kind}:${model.key}`)}
        />
      ))}

      {dialog === "quantization" && (
        <QuantizationDialog
          model={model}
          override={override}
          anyBusy={anyBusy}
          job={job}
          jobActive={jobActive}
          cancelBusy={cancelBusy}
          onCancelJob={onCancelJob}
          busy={busy}
          stateOf={stateOf}
          dismiss={dismiss}
          bits={bits}
          onBits={onBits}
          onConvert={onConvert}
          onActivate={onActivate}
          onPatch={onPatch}
          onClose={() => setDialog(null)}
        />
      )}

      {dialog === "settings" && (
        <ModelSettings
          model={model}
          caps={caps}
          override={override}
          busy={busy(`settings:${model.key}`)}
          disabled={anyBusy}
          note={stateOf(`settings:${model.key}`)}
          onDismissNote={() => dismiss(`settings:${model.key}`)}
          onApply={(patch) =>
            onPatch(patch, `settings:${model.key}`, "Saved. It applies when the server restarts.")
          }
          onClose={() => setDialog(null)}
        />
      )}
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
 * A dialog rather than a disclosure inside the row. Expanded in place, a form
 * this tall pushed every model below it off the screen, and the row it belonged
 * to became indistinguishable from the form's own header.
 *
 * Quantization is not here: it is two mechanisms with their own dialog, and a
 * third control called "Quantization" beside Steps was the ambiguity that made
 * a runtime setting look like it produced the saved copies listed one row up.
 *
 * Which controls are offered, and which are inert, is the backend's declaration —
 * `supports_guidance`, `supports_edit`. Nothing here decides what a model can do.
 */
function ModelSettings({
  model,
  caps,
  override,
  busy,
  disabled,
  note,
  onDismissNote,
  onApply,
  onClose,
}: {
  model: ModelStatus;
  caps: ModelCapabilities | undefined;
  override: Record<string, any>;
  busy: boolean;
  disabled: boolean;
  note: Parameters<typeof ActionNote>[0]["state"];
  onDismissNote: () => void;
  onApply: (patch: Record<string, unknown>) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const value = (key: string) => (key in draft ? draft[key] : override[key]);
  const set = (key: string, next: unknown) => setDraft((previous) => ({ ...previous, [key]: next }));
  const dirty = Object.keys(draft).length > 0;

  const numeric = (key: string) => ({
    type: "number" as const,
    value: (value(key) ?? "") as string | number,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      set(key, event.target.value === "" ? null : Number(event.target.value)),
  });

  // Off because this model has no edit path, as opposed to off because it is
  // switched off: the second is ours to change, the first is not. A stale `true`
  // in the configuration stays operable, so it can be switched back off.
  const editingUnsupported = caps ? !caps.supports_edit && override.enable_edit !== true : false;

  return (
    <Modal
      title={`Model settings - ${model.display_name}`}
      subtitle={model.repo}
      onClose={onClose}
    >
      <div className="setting-pair">
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
          <p className="setting-help">Denoising steps when a request names none.</p>
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
      </div>

      {/* A full-width row of its own, not a cell in the grid above. Sharing the
          narrow numeric columns broke "Expose the edits endpoint" over three
          lines beside a checkbox, and a wrapped label reads as a paragraph
          rather than as the name of the control next to it. */}
      <div className="setting">
        <span className="setting-label">Image editing</span>
        <div className="switch-row">
          <span className="switch-row-label">Expose the edits endpoint</span>
          <Switch
            checked={value("enable_edit") === true}
            label={`Expose the edits endpoint for ${model.key}`}
            disabled={editingUnsupported}
            onChange={(next) => set("enable_edit", next)}
          />
        </div>
        <p className="setting-help">
          {caps?.supports_edit === false
            ? "Not supported by this model: it has no image-editing path, so the endpoint would reject every request."
            : "Serves this model at /v1/images/edits as well as /v1/images/generations."}
        </p>
      </div>

      <div className="actions" style={{ marginTop: 16 }}>
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
      <ActionNote state={note} onDismiss={onDismissNote} />
    </Modal>
  );
}
