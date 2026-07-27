import { useEffect, useState } from "react";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import type { Capabilities } from "../types";

/**
 * A form over `server-config.json`.
 *
 * We edit a clone of the JSON rather than a TypeScript-typed model: the schema is
 * authoritative on the Python side, and duplicating it here would let it drift.
 * The server's capabilities (`/v1/capabilities`), on the other hand, are used to
 * grey out what makes no sense for a given model — the server already rejects
 * those values with a 400, so there is no point offering them.
 */
type Json = Record<string, any>;

const QUANTIZE_CHOICES = [null, 0, 3, 4, 5, 6, 8];

/**
 * Check a `WxH` resolution the way the server does.
 *
 * Worth checking here even though the server is the authority: an invalid
 * `default_size` makes `load_settings` refuse to start, so without this the
 * mistake would only surface as a failed launch later on — `fatal` is what blocks
 * the save. A truncation to a multiple of 16, in contrast, is legal and silent
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
  client,
  serverRunning,
  onSaved,
  onError,
}: {
  config: unknown;
  client: ServerClient | null;
  serverRunning: boolean;
  onSaved: () => void | Promise<void>;
  onError: (message: string) => void;
}) {
  const [draft, setDraft] = useState<Json | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [dirty, setDirty] = useState(false);

  // Never reseed over edits in flight. The root cause was `config` being re-read
  // on a timer, which is fixed upstream; this guard is what keeps the whole class
  // of bug from coming back through another door — any future caller that hands
  // us a fresh config object cannot silently discard what the user typed.
  useEffect(() => {
    if (dirty) return;
    setDraft(config ? (structuredClone(config) as Json) : null);
  }, [config, dirty]);

  useEffect(() => {
    if (!client) return;
    void client.capabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, [client]);

  if (!draft) return <p className="center-note">Loading the configuration…</p>;

  const server: Json = draft.server ?? {};
  const models: Json = draft.models ?? {};
  const sizeError = sizeProblem(String(draft.default_size ?? ""));

  function edit(next: Json) {
    setDraft(next);
    setDirty(true);
    setSaved(false);
  }

  function patchRoot(key: string, value: unknown) {
    edit({ ...draft, [key]: value });
  }

  function patchServer(key: string, value: unknown) {
    edit({ ...draft, server: { ...server, [key]: value } });
  }

  function patchModel(key: string, field: string, value: unknown) {
    edit({
      ...draft,
      models: { ...models, [key]: { ...(models[key] ?? {}), [field]: value } },
    });
  }

  async function save() {
    if (sizeError?.fatal) return;
    setSaving(true);
    try {
      await api.configWrite(draft);
      // Clearing `dirty` lets the reseed effect pick up what was just written;
      // `saved` survives it and is only cleared by the next edit.
      setDirty(false);
      setSaved(true);
      await onSaved();
    } catch (cause) {
      onError(messageOf(cause));
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <div className="card">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Server</h2>
          <div className="row">
            {saved && <span className="badge ok">saved</span>}
            <button
              className="primary"
              onClick={() => void save()}
              disabled={saving || sizeError?.fatal === true}
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
        <p className="hint">
          The configuration is only read at startup, so once saved{" "}
          {serverRunning
            ? "restart the server from the dashboard"
            : "it will apply on the next start"}
          .
        </p>

        <label className="field">
          <span>Default model</span>
          <select
            value={String(draft.default_model ?? "")}
            onChange={(event) => patchRoot("default_model", event.target.value)}
          >
            {Object.keys(models).map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Port</span>
          <input
            type="number"
            min={1}
            max={65535}
            value={Number(server.port ?? 8765)}
            onChange={(event) => patchServer("port", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Default resolution</span>
          <input
            type="text"
            placeholder="each model's own default"
            value={String(draft.default_size ?? "")}
            onChange={(event) => patchRoot("default_size", event.target.value || null)}
          />
        </label>
        {sizeError && (
          <p
            className="hint"
            style={{ marginTop: 0, color: `var(--${sizeError.fatal ? "error" : "warn"})` }}
          >
            {sizeError.message}
          </p>
        )}

        <label className="field">
          <span>Maximum images (n)</span>
          <input
            type="number"
            min={1}
            max={32}
            value={Number(server.max_n ?? 4)}
            onChange={(event) => patchServer("max_n", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Request timeout (s)</span>
          <input
            type="number"
            min={1}
            value={Number(server.request_timeout_s ?? 2400)}
            onChange={(event) => patchServer("request_timeout_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Graceful shutdown (s)</span>
          <input
            type="number"
            min={1}
            value={Number(server.shutdown_grace_s ?? 10)}
            onChange={(event) => patchServer("shutdown_grace_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>API key</span>
          <input
            type="password"
            placeholder="none"
            value={String(server.api_key ?? "")}
            onChange={(event) => patchServer("api_key", event.target.value || null)}
          />
        </label>

        <label className="field">
          <span>Image lifetime (s)</span>
          <input
            type="number"
            min={0}
            value={Number(server.image_ttl_s ?? 3600)}
            onChange={(event) => patchServer("image_ttl_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Log level</span>
          <select
            value={String(server.log_level ?? "INFO")}
            onChange={(event) => patchServer("log_level", event.target.value)}
          >
            {["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
        </label>

        <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
          The resolution applies to every model; a model pinned in the JSON with its own
          <code> default_size</code> still wins over it. The app sets the host, image directory and
          log format itself: those belong to how it operates, so they are not editable here.
        </p>
      </div>

      <div className="card">
        <h2>Models</h2>
        <p className="hint">
          Controls that do not apply are disabled based on the capabilities the server declares.
        </p>
        <table className="models">
          <thead>
            <tr>
              <th>Model</th>
              <th>On</th>
              <th>Quant.</th>
              <th>Steps</th>
              <th>Guidance</th>
              <th>Editing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(models).map(([key, raw]) => {
              const entry: Json = raw ?? {};
              const caps = capabilities?.models[key];
              return (
                <tr key={key}>
                  <td>
                    <strong>{key}</strong>
                    {caps && <div className="path">{caps.repo}</div>}
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={entry.enabled !== false}
                      onChange={(event) => patchModel(key, "enabled", event.target.checked)}
                    />
                  </td>
                  <td>
                    <select
                      value={entry.quantize === null || entry.quantize === undefined ? "" : String(entry.quantize)}
                      onChange={(event) =>
                        patchModel(key, "quantize", event.target.value === "" ? null : Number(event.target.value))
                      }
                    >
                      {QUANTIZE_CHOICES.map((bits) => (
                        <option key={String(bits)} value={bits === null ? "" : String(bits)}>
                          {bits === null ? "default" : bits === 0 ? "none (bf16)" : `${bits} bits`}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <input
                      type="number"
                      min={1}
                      style={{ width: 74 }}
                      placeholder={caps ? String(caps.default_steps) : "default"}
                      value={entry.default_steps ?? ""}
                      onChange={(event) =>
                        patchModel(key, "default_steps", event.target.value === "" ? null : Number(event.target.value))
                      }
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      min={0}
                      step={0.5}
                      style={{ width: 74 }}
                      // Distilled model: the server rejects any value.
                      disabled={caps ? !caps.supports_guidance : false}
                      placeholder={
                        caps?.supports_guidance === false
                          ? `fixed ${caps.default_guidance ?? 0}`
                          : caps
                            ? String(caps.default_guidance ?? "")
                            : "default"
                      }
                      value={entry.default_guidance ?? ""}
                      onChange={(event) =>
                        patchModel(
                          key,
                          "default_guidance",
                          event.target.value === "" ? null : Number(event.target.value),
                        )
                      }
                    />
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      disabled={caps ? !caps.supports_edit && entry.enable_edit !== true : false}
                      checked={entry.enable_edit === true}
                      onChange={(event) => patchModel(key, "enable_edit", event.target.checked)}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
