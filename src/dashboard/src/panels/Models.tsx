import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api";
import { PathPrompt } from "../PathPrompt";
import { messageOf } from "../api";
import { ActionNote, useActions } from "../actions";
import { describeJob, describeOutcome, type JobView } from "../job";
import { CatalogueGroup } from "./models/CatalogueGroup";
import { ImportConfirmation } from "./models/ImportConfirmation";
import { LocateConfirmation } from "./models/LocateConfirmation";
import { ModelRow } from "./models/ModelRow";
import { OPEN_TAB, CATALOGUE_TABS, lastChoice, slugFor } from "./models/shared";
import type { CatalogueTab, RowProps } from "./models/shared";
import type {
  Capabilities,
  ConfigWarning,
  ImportVerdict,
  JobStatus,
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
