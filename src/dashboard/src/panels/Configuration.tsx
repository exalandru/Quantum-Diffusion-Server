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
  effectiveHfHome,
  defaultCacheDir,
  hfTokenPresent,
  adminPasswordSet,
  lanAddresses,
  onSaved,
}: {
  config: unknown;
  /** Where weights are actually being read from right now. */
  effectiveHfHome: string | null;
  /** Where generated artifacts go when nothing is configured. Derived by the server
      from the application's data directory, so the placeholder is the real
      path rather than a description of one. */
  defaultCacheDir: string;
  /** Whether a token exists where `hf auth login` writes it. Not its value. */
  hfTokenPresent: boolean;
  /** Whether the control plane is protected. The network toggle requires it. */
  adminPasswordSet: boolean;
  /** This machine's own addresses, to show once it listens on the network. */
  lanAddresses: string[];
  onSaved: () => void | Promise<void>;
}) {
  const [draft, setDraft] = useState<Json | null>(null);
  const { run, dismiss, stateOf, busy } = useActions();
  const [dirty, setDirty] = useState(false);
  // Deliberately not part of the configuration draft: the token is not written to
  // `server-config.json` at all. It goes where `hf auth login` puts it, so a
  // secret that already sits there in plaintext is not duplicated into a second
  // file, and it is saved on its own rather than by the form's Save button.
  const [token, setToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  // The catalogue, so the rows below are the models the backend actually has.
  // They used to be the config file's key set, which meant a model added in a new
  // release stayed invisible — and unconfigurable — on every existing install.
  // Written by the server, into the same file `hf auth login` writes.
  const [catalogue, setCatalogue] = useState<ModelStatus[] | null>(null);

  useEffect(() => {
    void api
      .modelsStatus()
      .then((status) => setCatalogue(status.models))
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

  /** Its own action, its own result: nothing about it belongs to the form's save. */
  async function saveToken() {
    await run(
      "token",
      async () => {
        await api.hfTokenWrite(token);
        setToken("");
        // So the "token present" state stops contradicting what was just saved.
        await onSaved();
      },
      "Token saved.",
    );
  }

  // No folder chooser: a web page has no access to one, and inventing a
  // directory-browsing endpoint would mean the server listing the filesystem to
  // anything that can reach it — a real surface, bought for a convenience. The
  // paths are typed instead, and the catalogue already reports an unusable one
  // (`cache_dir_unavailable`, `hf_home_unavailable`) rather than failing later.

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
      "Saved. Restart the server from the Dashboard to apply it.",
    );
  }

  // The host *is* the state: a second field describing it would be a second
  // source of truth, and they would disagree the first time anyone edited the
  // file by hand.
  async function saveAdminPassword() {
    await run(
      "password",
      async () => {
        await api.setPassword(newPassword.trim(), currentPassword.trim() || undefined);
        setNewPassword("");
        setCurrentPassword("");
        // Every session was minted against the old password, including this
        // page's. Refreshing is what turns that into a login screen rather than
        // a screen that quietly stops working.
        await onSaved();
      },
      "Password saved. Every browser has been signed out.",
    );
  }

  const lanEnabled = String(server.host ?? "127.0.0.1") === "0.0.0.0";
  const canEnableLan = adminPasswordSet && Boolean(server.api_key);
  const lanBlockedReason = !adminPasswordSet
    ? "Set an admin password first: the control plane must not be reachable from the network without one."
    : "Set an API key first: /v1 must not be reachable from the network without one.";

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
          The generation server reads this file only at startup, so a saved change applies when
          you restart it from the Dashboard. Model management — the catalogue, downloads, imports,
          conversions — reads it immediately.
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
            The app sets the image directory and log format itself: those belong to how it
            operates, so they are not editable here.
          </p>

          <hr className="setting-divider" />

          {/* Its own control with its own button, not a form field, because it
              is not part of the document this form saves: the password is
              hashed into a separate file, so there is nothing here to round-trip
              and Save has no business touching it. Same shape as the Hugging
              Face token below, for the same reason. */}
          <div className="setting">
            <span className="setting-label" id="admin-password-label">
              Admin password
            </span>
            <div className="setting-row">
              <span className={adminPasswordSet ? "pill pill-ok" : "pill pill-warn"}>
                {adminPasswordSet ? "password set" : "no password"}
              </span>
              {adminPasswordSet && (
                <input
                  type="password"
                  aria-label="Current admin password"
                  placeholder="current"
                  autoComplete="current-password"
                  value={currentPassword}
                  onChange={(event) => setCurrentPassword(event.target.value)}
                />
              )}
              <input
                type="password"
                aria-labelledby="admin-password-label"
                placeholder={adminPasswordSet ? "new password" : "at least 8 characters"}
                autoComplete="new-password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
              />
              <button
                onClick={() => void saveAdminPassword()}
                disabled={newPassword.trim().length < 8 || busy("password")}
              >
                {busy("password") ? "Saving…" : adminPasswordSet ? "Change" : "Set password"}
              </button>
            </div>
            <ActionNote state={stateOf("password")} onDismiss={() => dismiss("password")} />
            <p className="setting-help">
              Protects this screen, the catalogue, the logs and the restart button. It is{" "}
              <strong>not</strong> the API key: that one is for <code>/v1</code>, so anything you
              point at this server to generate images cannot also reconfigure it. Changing the
              password signs every browser out, including this one.
            </p>
            {!adminPasswordSet && (
              <p className="setting-help">
                Without one, the control plane is open to anything that can reach this port — which
                is only safe because the server refuses to listen beyond this machine until a
                password exists.
              </p>
            )}
          </div>

          <hr className="setting-divider" />

          {/* A boolean that is a sentence, so it gets a row rather than a cell
              in the four-column grid — the same shape every other switch in the
              app uses. */}
          <div className="switch-row">
            <span className="switch-row-label" id="lan-label">
              Reachable from the local network
            </span>
            <button
              role="switch"
              className="switch"
              aria-checked={lanEnabled}
              aria-labelledby="lan-label"
              disabled={!lanEnabled && !canEnableLan}
              title={!lanEnabled && !canEnableLan ? lanBlockedReason : undefined}
              onClick={() => patchServer("host", lanEnabled ? "127.0.0.1" : "0.0.0.0")}
            />
          </div>

          {/* Refused rather than allowed-and-warned: the server will not start
              bound to the network without both credentials, so offering the
              switch would only produce a configuration that cannot come back. */}
          {!lanEnabled && !canEnableLan && (
            <p className="setting-help">{lanBlockedReason}</p>
          )}

          {lanEnabled && (
            <>
              <p className="caution">
                Anyone on this network can reach this server. The connection is plain HTTP: the
                admin password, the session cookie and the API key are sent unencrypted and can be
                read by anyone on the same network. Use this only on a network you trust.
              </p>
              {lanAddresses.length > 0 && (
                <p className="setting-help">
                  Reachable at{" "}
                  {lanAddresses.map((address, index) => (
                    <span key={address}>
                      {index > 0 && ", "}
                      <code>
                        http://{address}:{String(server.port ?? 8765)}/dashboard
                      </code>
                    </span>
                  ))}
                  . Point OpenAI clients at <code>/v1</code> on the same address.
                </p>
              )}
            </>
          )}
        </fieldset>

        {/* ── Hugging Face ────────────────────────────────────────────── */}
        {/* Where models come from and what proves we may have them: one source,
            configured once for the application. Both used to live in Models —
            the token beside the catalogue, the folder here — which put an
            account-wide secret among per-model controls and split one subject
            across two views. */}
        <fieldset className="settings-group">
          <legend>Hugging Face</legend>

          <div className="setting">
            {/* The field names itself — "Hugging Face token" rather than
                "Access token" — because a label queried on its own has to say
                which token it is. This heading is for the eye. */}
            <span className="setting-label">Access token</span>
            <div className="setting-row">
              <span className={hfTokenPresent ? "pill pill-ok" : "pill pill-warn"}>
                {hfTokenPresent ? "token present" : "no token"}
              </span>
              <input
                type="password"
                aria-label="Hugging Face token"
                placeholder="hf_…"
                value={token}
                onChange={(event) => {
                  setToken(event.target.value);
                  // Typing a new token supersedes the previous save's result.
                  dismiss("token");
                }}
              />
              <button
                onClick={() => void saveToken()}
                disabled={token.trim().length === 0 || busy("token")}
              >
                {busy("token") ? "Saving…" : "Save token"}
              </button>
            </div>
            <ActionNote state={stateOf("token")} onDismiss={() => dismiss("token")} />
            <p className="setting-help">
              Gated repositories need a token whose account has been granted access on each model's
              card. It is stored where <code>hf auth login</code> writes it, so as not to duplicate a
              secret that already sits there in plaintext, and it is never copied into this
              configuration file.
            </p>
            <code className="library-path">{effectiveHfHome ?? "~/.cache/huggingface"}/token</code>
          </div>

          {/* Two settings, one subject. What proves we may fetch weights and
              where they land are related enough to share a section and separate
              enough that running them together reads as one long form — so they
              get the rule this interface already uses between sections, rather
              than a second card that would deny the relationship. */}
          <hr className="setting-divider" />

          <div className="setting">
            {/* Not a <label>: it wraps buttons, and a label both mis-targets
                clicks and lends its own text to the buttons' accessible names. */}
            <span className="setting-label" id="storage-label">
              Hugging Face model directory
            </span>
            <div className="setting-row">
              <input
                type="text"
                aria-labelledby="storage-label"
                spellCheck={false}
                value={String(storage.hf_home ?? "")}
                placeholder={effectiveHfHome ?? "default (~/.cache/huggingface)"}
                onChange={(event) => patchStorage("hf_home", event.target.value || null)}
              />
              {storage.hf_home && (
                <button onClick={() => patchStorage("hf_home", null)}>Use default</button>
              )}
            </div>
            <p className="setting-help">
              Where weights are downloaded to and discovered from — an external SSD works. Changing
              it moves nothing: the previous folder is left exactly as it is, so models that live
              only there stop being listed until you point back at it. A folder on a volume that is
              not mounted is reported as unavailable rather than treated as empty.
            </p>
            <p className="caution">
              The running server keeps the folder it was launched with until you restart it. Model
              management already uses the new one.
            </p>
          </div>
        </fieldset>

        {/* ── Storage ─────────────────────────────────────────────────── */}
        {/* A separate section from Hugging Face, because these are two
            different kinds of file with two different lifetimes: one holds
            weights that can be downloaded again, the other holds hours of
            conversion that cannot. Either can live on its own disk. */}
        <fieldset className="settings-group">
          <legend>Storage</legend>

          <div className="setting">
            {/* Not a <label>: it wraps buttons, and a label both mis-targets
                clicks and lends its own text to the buttons' accessible names. */}
            <span className="setting-label" id="cache-label">
              Pre-quantized model cache
            </span>
            <div className="setting-row">
              <input
                type="text"
                aria-labelledby="cache-label"
                spellCheck={false}
                value={String(storage.cache_dir ?? "")}
                placeholder={defaultCacheDir}
                onChange={(event) => patchStorage("cache_dir", event.target.value || null)}
              />
              {storage.cache_dir && (
                <button onClick={() => patchStorage("cache_dir", null)}>Use default</button>
              )}
            </div>
            <p className="setting-help">
              Where QDS stores the pre-quantized copies it generates, with their completion records
              and component progress. Changing it moves nothing: existing copies stay where they
              are, and only new conversions — and what the Models tab finds — follow the new folder.
              A folder on a volume that is not mounted is reported as unavailable rather than
              treated as empty.
            </p>
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
          What is here is what applies to every model at once.
        </p>
      </section>
    </>
  );
}
