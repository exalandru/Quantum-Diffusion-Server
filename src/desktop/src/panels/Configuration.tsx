import { useEffect, useState } from "react";

import * as api from "../api";
import { ActionNote, useActions } from "../actions";
import type { ModelStatus } from "../types";

/**
 * A form over `server-config.json`, grouped by what each setting is about.
 *
 * We edit a clone of the JSON rather than a TypeScript-typed model: the schema is
 * authoritative on the Python side, and duplicating it here would let it drift.
 * The server's capabilities (`/v1/capabilities`) are used only to grey out what
 * makes no sense for a given model — the server already rejects those values with
 * a 400, so offering them buys nothing.
 */
type Json = Record<string, any>;

/**
 * Check a `WxH` resolution the way the server does.
 *
 * Worth checking here even though the server is the authority: an invalid
 * `default_size` makes `load_settings` refuse to start, so without this the
 * mistake would only surface as a failed launch later. `fatal` is what blocks the
 * save. A truncation to a multiple of 16, in contrast, is legal and silent
 * server-side; we only say so rather than let it surprise them.
 */
function sizeProblem(value: string): { message: string; fatal: boolean } | null {
  if (value.trim() === "") return null;
  const match = /^(\d+)\s*[xX]\s*(\d+)$/.exec(value.trim());
  if (!match) return { message: "Expected WxH, for example 1024x1024.", fatal: true };
  const [width, height] = [Number(match[1]), Number(match[2])];
  if (width < 16 || height < 16) return { message: "Each side must be at least 16.", fatal: true };
  if (width % 16 || height % 16) {
    const [w, h] = [16 * Math.floor(width / 16), 16 * Math.floor(height / 16)];
    return { message: `Will be truncated to ${w}x${h}.`, fatal: false };
  }
  return null;
}

export function Configuration({
  config,
  serverRunning,
  effectiveHfHome,
  onSaved,
}: {
  config: unknown;
  serverRunning: boolean;
  /** Where weights are actually being read from right now. */
  effectiveHfHome: string | null;
  onSaved: () => void | Promise<void>;
}) {
  const [draft, setDraft] = useState<Json | null>(null);
  const { run, dismiss, stateOf, busy } = useActions();
  const [dirty, setDirty] = useState(false);
  // The catalogue, so the rows below are the models the backend actually has.
  // They used to be the config file's key set, which meant a model added in a new
  // release stayed invisible — and unconfigurable — on every existing install.
  // Comes through Rust, so it still works with the generation server stopped.
  const [catalogue, setCatalogue] = useState<ModelStatus[] | null>(null);

  useEffect(() => {
    void api
      .modelsStatus()
      .then(setCatalogue)
      .catch(() => setCatalogue(null));
  }, []);

  // Never reseed over edits in flight. The root cause was `config` being re-read
  // on a timer, which is fixed upstream; this guard keeps the whole class of bug
  // from coming back through another door.
  useEffect(() => {
    if (dirty) return;
    setDraft(config ? (structuredClone(config) as Json) : null);
  }, [config, dirty]);

  if (!draft) return <p className="empty">Loading the configuration…</p>;

  const server: Json = draft.server ?? {};
  const storage: Json = draft.storage ?? {};
  const models: Json = draft.models ?? {};
  // Catalogue first, config as overrides layered on top. Falling back to the
  // config's own keys keeps the form usable if the catalogue cannot be read.
  const modelKeys: string[] = catalogue ? catalogue.map((row) => row.key) : Object.keys(models);
  const rowOf = (key: string) => catalogue?.find((row) => row.key === key);
  const sizeError = sizeProblem(String(draft.default_size ?? ""));

  // The server refuses a configuration whose default model is disabled, and
  // refuses it at *startup* — so without this the mistake is saved silently and
  // only surfaces as "Invalid configuration" on the next launch. The switch that
  // causes it now lives on the model's row, so that is where the message points.
  const disabledDefault =
    models[String(draft.default_model ?? "")]?.enabled === false
      ? String(draft.default_model)
      : null;
  const blocked = sizeError?.fatal === true || disabledDefault !== null;

  function edit(next: Json) {
    setDraft(next);
    setDirty(true);
    // The owning component superseding its own result: a new edit makes the
    // previous save's outcome stale. Nothing else may clear it.
    dismiss("save");
  }

  const patchRoot = (key: string, value: unknown) => edit({ ...draft, [key]: value });
  const patchServer = (key: string, value: unknown) =>
    edit({ ...draft, server: { ...server, [key]: value } });
  const patchStorage = (key: string, value: unknown) =>
    edit({ ...draft, storage: { ...storage, [key]: value } });

  async function chooseFolder() {
    await run("storage", async () => {
      const chosen = await api.pickDirectory(String(storage.hf_home ?? effectiveHfHome ?? ""));
      // Cancelling is not a failure, and must not look like one.
      if (chosen) patchStorage("hf_home", chosen);
    });
  }

  /**
   * Save the sections this view owns, over whatever is on disk now.
   *
   * Not `configWrite(draft)`: the draft is a clone taken when the form was
   * seeded, and it carries a `models` section this view no longer edits. Writing
   * it back wholesale would undo any switch flipped in Models while this form sat
   * open — a real sequence, because the two views are one click apart. Re-reading
   * first and splicing in only `default_model`, `default_size`, `server` and
   * `storage` makes the ownership split actual rather than nominal.
   */
  async function save() {
    if (blocked || !draft) return;
    await run(
      "save",
      async () => {
        const current = ((await api.configRead()) ?? {}) as Json;
        await api.configWrite({
          ...current,
          default_model: draft.default_model,
          default_size: draft.default_size,
          server: draft.server,
          storage: draft.storage,
        });
        // Clearing `dirty` lets the reseed effect pick up what was just written;
        // the saved result survives it and is only cleared by the next edit.
        setDirty(false);
        await onSaved();
      },
      serverRunning
        ? "Saved. Restart the server from the Dashboard to apply it."
        : "Saved. It will apply on the next start.",
    );
  }

  const number = (key: string, value: unknown, patch: (key: string, value: unknown) => void) => ({
    type: "number" as const,
    value: (value ?? "") as string | number,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      patch(key, event.target.value === "" ? null : Number(event.target.value)),
  });

  return (
    <>
      <section className="panel">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Configuration</h2>
          <div className="row">
            <button
              className="primary"
              onClick={() => void save()}
              disabled={busy("save") || blocked}
              title={blocked ? "Fix the highlighted problem before saving." : undefined}
            >
              {busy("save") ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
        {/* Truthful restart semantics, stated once and near the button that
            creates the situation. Nothing here restarts anything by itself. */}
        <p className="note">
          The generation server reads this file only at startup, so a saved change{" "}
          {serverRunning
            ? "applies when you restart it from the Dashboard."
            : "applies on the next start."}{" "}
          Model management — the catalogue, downloads, imports, conversions — reads it immediately.
        </p>
        <ActionNote state={stateOf("save")} onDismiss={() => dismiss("save")} />

        {/* ── Generation defaults ─────────────────────────────────────── */}
        <fieldset className="settings-group">
          <legend>Generation defaults</legend>

          <div className="setting-pair">
            <div className="setting">
              <label className="setting-label" htmlFor="default-model">
                Default model
              </label>
              <select
                id="default-model"
                value={String(draft.default_model ?? "")}
                onChange={(event) => patchRoot("default_model", event.target.value)}
              >
                {modelKeys.map((key) => (
                  <option key={key} value={key}>
                    {rowOf(key)?.display_name ?? key}
                  </option>
                ))}
              </select>
              <p className="setting-help">Used when a request names no model.</p>
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="default-size">
                Default resolution
              </label>
              <input
                id="default-size"
                type="text"
                placeholder="each model's own default"
                value={String(draft.default_size ?? "")}
                onChange={(event) => patchRoot("default_size", event.target.value || null)}
              />
              {sizeError ? (
                <p className={sizeError.fatal ? "setting-error" : "caution"}>{sizeError.message}</p>
              ) : (
                <p className="setting-help">
                  Applies to every model; a model pinned with its own size still wins.
                </p>
              )}
            </div>
          </div>
        </fieldset>

        {/* ── Server ──────────────────────────────────────────────────── */}
        <fieldset className="settings-group">
          <legend>Server</legend>

          <div className="setting-pair">
            <div className="setting">
              <label className="setting-label" htmlFor="port">
                Port
              </label>
              <input id="port" min={1} max={65535} {...number("port", server.port ?? 8765, patchServer)} />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="api-key">
                API key
              </label>
              <input
                id="api-key"
                type="password"
                placeholder="none"
                value={String(server.api_key ?? "")}
                onChange={(event) => patchServer("api_key", event.target.value || null)}
              />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="max-n">
                Maximum images per request
              </label>
              <input id="max-n" min={1} max={32} {...number("max_n", server.max_n ?? 4, patchServer)} />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="timeout">
                Request timeout <span className="setting-unit">seconds</span>
              </label>
              <input
                id="timeout"
                min={1}
                {...number("request_timeout_s", server.request_timeout_s ?? 2400, patchServer)}
              />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="grace">
                Graceful shutdown <span className="setting-unit">seconds</span>
              </label>
              <input
                id="grace"
                min={1}
                {...number("shutdown_grace_s", server.shutdown_grace_s ?? 10, patchServer)}
              />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="ttl">
                Image lifetime <span className="setting-unit">seconds</span>
              </label>
              <input id="ttl" min={0} {...number("image_ttl_s", server.image_ttl_s ?? 3600, patchServer)} />
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="idle">
                Free memory after <span className="setting-unit">seconds</span>
              </label>
              <input
                id="idle"
                min={0}
                placeholder="never"
                {...number("idle_unload_s", server.idle_unload_s, patchServer)}
              />
              <p className="setting-help">
                Releases the warm model after that long without a generation, so something else can
                have the memory. Empty keeps it warm forever, <code>0</code> frees it as soon as the
                request ends. The cost is paying the load again on the next image.
              </p>
            </div>

            <div className="setting">
              <label className="setting-label" htmlFor="log-level">
                Log level
              </label>
              <select
                id="log-level"
                value={String(server.log_level ?? "INFO")}
                onChange={(event) => patchServer("log_level", event.target.value)}
              >
                {["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <p className="setting-help">
            The app sets the host, image directory and log format itself: those belong to how it
            operates, so they are not editable here.
          </p>
        </fieldset>

        {/* ── Storage ─────────────────────────────────────────────────── */}
        <fieldset className="settings-group">
          <legend>Storage</legend>

          <div className="setting">
            {/* Not a <label>: it wraps buttons, and a label both mis-targets
                clicks and lends its own text to the buttons' accessible names. */}
            <span className="setting-label" id="storage-label">
              Hugging Face model directory
            </span>
            <div className="setting-row">
              <input
                type="text"
                readOnly
                aria-labelledby="storage-label"
                value={String(storage.hf_home ?? "")}
                placeholder={effectiveHfHome ?? "default (~/.cache/huggingface)"}
              />
              <button onClick={() => void chooseFolder()} disabled={busy("storage")}>
                {busy("storage") ? "Choosing…" : "Choose Folder…"}
              </button>
              {storage.hf_home && (
                <button onClick={() => patchStorage("hf_home", null)}>Use default</button>
              )}
            </div>
            <ActionNote state={stateOf("storage")} onDismiss={() => dismiss("storage")} />
            <p className="setting-help">
              Where weights are downloaded to and discovered from — an external SSD works. Changing
              it moves nothing: the previous folder is left exactly as it is, so models that live
              only there stop being listed until you point back at it. A folder on a volume that is
              not mounted is reported as unavailable rather than treated as empty.
            </p>
            {serverRunning && (
              <p className="caution">
                The running server keeps the folder it was launched with until you restart it. Model
                management already uses the new one.
              </p>
            )}
          </div>
        </fieldset>

        {/* Per-model settings used to sit here as a second table, which meant a
            model appeared twice under two names: its availability and conversion
            in Models, its `enabled`/quantize/steps in a form here. They now live
            on the model's own row, which is also where you turn it on and
            download it. */}
        <p className="setting-help" style={{ marginTop: 16 }}>
          Per-model settings — enabled, quantization, steps, guidance, editing — live on each
          model's row in the <strong>Models</strong> view, beside its weights and saved variants.
        </p>
      </section>
    </>
  );
}
