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
  onSaved: () => void;
  onError: (message: string) => void;
}) {
  const [draft, setDraft] = useState<Json | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setDraft(config ? (structuredClone(config) as Json) : null);
    setSaved(false);
  }, [config]);

  useEffect(() => {
    if (!client) return;
    void client.capabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, [client]);

  if (!draft) return <p className="center-note">Loading the configuration…</p>;

  const server: Json = draft.server ?? {};
  const models: Json = draft.models ?? {};

  function patchServer(key: string, value: unknown) {
    setDraft({ ...draft, server: { ...server, [key]: value } });
    setSaved(false);
  }

  function patchModel(key: string, field: string, value: unknown) {
    setDraft({
      ...draft,
      models: { ...models, [key]: { ...(models[key] ?? {}), [field]: value } },
    });
    setSaved(false);
  }

  async function save() {
    setSaving(true);
    try {
      await api.configWrite(draft);
      setSaved(true);
      onSaved();
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
            <button className="primary" onClick={() => void save()} disabled={saving}>
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
            onChange={(event) => {
              setDraft({ ...draft, default_model: event.target.value });
              setSaved(false);
            }}
          >
            {Object.keys(models).map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </label>

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
          The app sets the host, port, image directory and log format itself: those values belong
          to how it operates, so they are not editable here.
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
