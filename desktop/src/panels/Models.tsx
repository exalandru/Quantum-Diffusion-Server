import { useEffect, useState } from "react";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import type { Capabilities, Overview } from "../types";

/** Ordre imposé par le script : le plus gros d'abord, pour borner le pic disque. */
const COMPONENTS = [
  { id: "transformer", label: "Transformer", detail: "64,5 Go en bf16 → environ 34 Go" },
  { id: "text_encoder", label: "Encodeur texte", detail: "45,8 Go en bf16 → environ 24 Go" },
  { id: "vae", label: "VAE", detail: "0,34 Go" },
];

export function Models({
  state,
  client,
  config,
  onError,
}: {
  state: Overview;
  client: ServerClient | null;
  config: unknown;
  onError: (message: string) => void;
}) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [token, setToken] = useState("");
  const [tokenSaved, setTokenSaved] = useState(false);
  const [selected, setSelected] = useState<string[]>(COMPONENTS.map((component) => component.id));
  const [converting, setConverting] = useState(false);

  useEffect(() => {
    if (!client) return;
    void client.capabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, [client]);

  const modelPath =
    (config as { models?: { "flux2-dev"?: { model_path?: string | null } } } | null)?.models?.[
      "flux2-dev"
    ]?.model_path ?? null;

  async function saveToken() {
    try {
      await api.hfTokenWrite(token);
      setTokenSaved(true);
      setToken("");
    } catch (cause) {
      onError(messageOf(cause));
    }
  }

  async function convert() {
    setConverting(true);
    try {
      await api.prequantizeRun(selected, modelPath ?? undefined);
    } catch (cause) {
      onError(messageOf(cause));
    } finally {
      setConverting(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2>Accès HuggingFace</h2>
        <p className="hint">
          Les dépôts <code>black-forest-labs/*</code> sont à accès restreint : il faut un token dont
          l'accès a été accordé. Il est enregistré là où <code>hf auth login</code> l'écrit, pour ne
          pas dupliquer un secret déjà présent en clair.
        </p>
        <div className="row">
          <span className={`badge ${state.hfTokenPresent ? "ok" : "warn"}`}>
            <span className="dot" />
            {state.hfTokenPresent ? "token présent" : "aucun token"}
          </span>
          <input
            type="password"
            placeholder="hf_…"
            value={token}
            onChange={(event) => {
              setToken(event.target.value);
              setTokenSaved(false);
            }}
            style={{ flex: 1, minWidth: 220 }}
          />
          <button onClick={() => void saveToken()} disabled={token.trim().length === 0}>
            Enregistrer
          </button>
          {tokenSaved && <span className="badge ok">enregistré</span>}
        </div>
        <p className="path" style={{ marginTop: 10 }}>
          {state.hfHome}/token
        </p>
      </div>

      <div className="card">
        <h2>FLUX.2 [dev]</h2>
        <p className="hint">
          Le seul modèle qui demande une préparation. Son dépôt est en bf16 : transformer 64,5 Go,
          encodeur texte 45,8 Go, soit environ <strong>111 Go de poids résidents</strong> — hors
          d'atteinte de la mémoire unifiée. En 8 bits on retombe à environ 58 Go, mais quantifier au
          chargement supposerait justement de tenir le bf16 en mémoire d'abord. D'où cette conversion,
          à faire une fois.
        </p>

        <div className="row" style={{ marginBottom: 12 }}>
          <span className={`badge ${state.flux2DevReady ? "ok" : "warn"}`}>
            <span className="dot" />
            {state.flux2DevReady ? "artefact présent" : "artefact absent"}
          </span>
          {!state.flux2DevReady && (
            <span className="hint" style={{ margin: 0 }}>
              Sans lui, le serveur répond 503 <code>model_not_prepared</code>.
            </span>
          )}
        </div>

        <fieldset>
          <legend>Composants à convertir</legend>
          {COMPONENTS.map((component) => (
            <label className="check" key={component.id}>
              <input
                type="checkbox"
                checked={selected.includes(component.id)}
                onChange={(event) =>
                  setSelected((previous) =>
                    event.target.checked
                      ? COMPONENTS.filter(
                          (item) => previous.includes(item.id) || item.id === component.id,
                        ).map((item) => item.id)
                      : previous.filter((id) => id !== component.id),
                  )
                }
              />
              <span>
                {component.label} <span className="hint">· {component.detail}</span>
              </span>
            </label>
          ))}
        </fieldset>

        <div className="row">
          <button
            className="primary"
            onClick={() => void convert()}
            disabled={converting || selected.length === 0}
          >
            {converting ? "Conversion en cours…" : "Lancer la conversion"}
          </button>
          <span className="hint" style={{ margin: 0 }}>
            Suis l'avancement dans l'onglet Logs.
          </span>
        </div>

        <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
          La conversion travaille composant par composant, et quantifie le transformer bloc par bloc :
          sans ça le pic mémoire atteindrait environ 96 Go, contre environ 66 ainsi. Entre deux
          composants, purger le bf16 du cache HuggingFace fait tomber le pic disque de 169 à environ
          97 Go — le script rappelle quoi supprimer.
        </p>
      </div>

      <div className="card">
        <h2>Catalogue</h2>
        {capabilities ? (
          <table className="models">
            <thead>
              <tr>
                <th>Modèle</th>
                <th>Taille</th>
                <th>Étapes</th>
                <th>Guidance</th>
                <th>Quant.</th>
                <th>Capacités</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(capabilities.models).map(([key, caps]) => (
                <tr key={key}>
                  <td>
                    <strong>{key}</strong>
                    {key === capabilities.default_model && (
                      <span className="badge" style={{ marginLeft: 6 }}>
                        défaut
                      </span>
                    )}
                    <div className="path">{caps.repo}</div>
                  </td>
                  <td>{caps.default_size}</td>
                  <td>{caps.default_steps}</td>
                  <td>
                    {caps.default_guidance ?? "—"}
                    {!caps.supports_guidance && <span className="hint"> figée</span>}
                  </td>
                  <td>{caps.quantize ? `${caps.quantize} bits` : "—"}</td>
                  <td>
                    <div className="row" style={{ gap: 5 }}>
                      {caps.supports_negative_prompt && <span className="badge">negatif</span>}
                      {caps.supports_image_to_image && <span className="badge">img2img</span>}
                      {caps.supports_edit && <span className="badge">édition</span>}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="hint" style={{ marginBottom: 0 }}>
            Démarre le serveur pour voir le catalogue et les capacités déclarées.
          </p>
        )}
      </div>
    </>
  );
}
