import { useEffect, useState } from "react";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import type { Capabilities } from "../types";

/**
 * Formulaire sur `server-config.json`.
 *
 * On édite un clone du JSON plutôt qu'un modèle typé en TypeScript : le schéma
 * fait autorité côté Python, et le dupliquer ici le laisserait dériver. Les
 * capacités du serveur (`/v1/capabilities`) servent en revanche à griser ce qui
 * n'a pas de sens pour un modèle donné — le serveur refuse déjà ces valeurs par
 * un 400, autant ne pas les proposer.
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

  if (!draft) return <p className="center-note">Chargement de la configuration…</p>;

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
          <h2 style={{ margin: 0 }}>Serveur</h2>
          <div className="row">
            {saved && <span className="badge ok">enregistré</span>}
            <button className="primary" onClick={() => void save()} disabled={saving}>
              {saving ? "Enregistrement…" : "Enregistrer"}
            </button>
          </div>
        </div>
        <p className="hint">
          La configuration n'est lue qu'au démarrage : après enregistrement,{" "}
          {serverRunning ? "redémarre le serveur depuis le tableau de bord" : "elle s'appliquera au prochain démarrage"}.
        </p>

        <label className="field">
          <span>Modèle par défaut</span>
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
          <span>Nombre d'images maximal (n)</span>
          <input
            type="number"
            min={1}
            max={32}
            value={Number(server.max_n ?? 4)}
            onChange={(event) => patchServer("max_n", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Délai maximal (s)</span>
          <input
            type="number"
            min={1}
            value={Number(server.request_timeout_s ?? 2400)}
            onChange={(event) => patchServer("request_timeout_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Arrêt gracieux (s)</span>
          <input
            type="number"
            min={1}
            value={Number(server.shutdown_grace_s ?? 10)}
            onChange={(event) => patchServer("shutdown_grace_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Clé d'API</span>
          <input
            type="password"
            placeholder="aucune"
            value={String(server.api_key ?? "")}
            onChange={(event) => patchServer("api_key", event.target.value || null)}
          />
        </label>

        <label className="field">
          <span>Durée de vie des images (s)</span>
          <input
            type="number"
            min={0}
            value={Number(server.image_ttl_s ?? 3600)}
            onChange={(event) => patchServer("image_ttl_s", Number(event.target.value))}
          />
        </label>

        <label className="field">
          <span>Niveau de log</span>
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
          L'app impose elle-même l'hôte, le port, le dossier d'images et le format des logs : ces
          valeurs sont propres à son fonctionnement et ne sont donc pas éditables ici.
        </p>
      </div>

      <div className="card">
        <h2>Modèles</h2>
        <p className="hint">
          Les contrôles inapplicables sont désactivés d'après les capacités déclarées par le serveur.
        </p>
        <table className="models">
          <thead>
            <tr>
              <th>Modèle</th>
              <th>Actif</th>
              <th>Quantification</th>
              <th>Étapes</th>
              <th>Guidance</th>
              <th>Édition</th>
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
                          {bits === null ? "défaut" : bits === 0 ? "aucune (bf16)" : `${bits} bits`}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <input
                      type="number"
                      min={1}
                      style={{ width: 74 }}
                      placeholder={caps ? String(caps.default_steps) : "défaut"}
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
                      // Modèle distillé : le serveur refuse toute valeur.
                      disabled={caps ? !caps.supports_guidance : false}
                      placeholder={
                        caps?.supports_guidance === false
                          ? `figée ${caps.default_guidance ?? 0}`
                          : caps
                            ? String(caps.default_guidance ?? "")
                            : "défaut"
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
