import { useEffect, useState } from "react";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import type { Capabilities, Overview } from "../types";

/** Order enforced by the script: biggest first, to bound the disk peak. */
const COMPONENTS = [
  { id: "transformer", label: "Transformer", detail: "64.5 GB in bf16 → about 34 GB" },
  { id: "text_encoder", label: "Text encoder", detail: "45.8 GB in bf16 → about 24 GB" },
  { id: "vae", label: "VAE", detail: "0.34 GB" },
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
        <h2>HuggingFace access</h2>
        <p className="hint">
          The <code>black-forest-labs/*</code> repos are gated: you need a token that has been
          granted access. It is stored where <code>hf auth login</code> writes it, so as not to
          duplicate a secret that already sits there in plaintext.
        </p>
        <div className="row">
          <span className={`badge ${state.hfTokenPresent ? "ok" : "warn"}`}>
            <span className="dot" />
            {state.hfTokenPresent ? "token present" : "no token"}
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
            Save
          </button>
          {tokenSaved && <span className="badge ok">saved</span>}
        </div>
        <p className="path" style={{ marginTop: 10 }}>
          {state.hfHome}/token
        </p>
      </div>

      <div className="card">
        <h2>FLUX.2 [dev]</h2>
        <p className="hint">
          The only model that needs preparation. Its repo ships bf16: a 64.5 GB transformer and a
          45.8 GB text encoder, so about <strong>111 GB of resident weights</strong> — out of reach
          for unified memory. At 8 bits we come back down to roughly 58 GB, but quantizing at load
          time would mean holding the bf16 in memory first. Hence this one-time conversion.
        </p>

        <div className="row" style={{ marginBottom: 12 }}>
          <span className={`badge ${state.flux2DevReady ? "ok" : "warn"}`}>
            <span className="dot" />
            {state.flux2DevReady ? "artifact present" : "artifact missing"}
          </span>
          {!state.flux2DevReady && (
            <span className="hint" style={{ margin: 0 }}>
              Without it, the server answers 503 <code>model_not_prepared</code>.
            </span>
          )}
        </div>

        <fieldset>
          <legend>Components to convert</legend>
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
            {converting ? "Converting…" : "Start conversion"}
          </button>
          <span className="hint" style={{ margin: 0 }}>
            Follow the progress in the Logs tab.
          </span>
        </div>

        <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
          The conversion works one component at a time, and quantizes the transformer block by
          block: without that the memory peak would reach about 96 GB, against roughly 66 this way.
          Between components, purging the bf16 from the HuggingFace cache brings the disk peak down
          from 169 to about 97 GB — the script reminds you what to delete.
        </p>
      </div>

      <div className="card">
        <h2>Catalogue</h2>
        {capabilities ? (
          <table className="models">
            <thead>
              <tr>
                <th>Model</th>
                <th>Size</th>
                <th>Steps</th>
                <th>Guidance</th>
                <th>Quant.</th>
                <th>Features</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(capabilities.models).map(([key, caps]) => (
                <tr key={key}>
                  <td>
                    <strong>{key}</strong>
                    {key === capabilities.default_model && (
                      <span className="badge" style={{ marginLeft: 6 }}>
                        default
                      </span>
                    )}
                    <div className="path">{caps.repo}</div>
                  </td>
                  <td>{caps.default_size}</td>
                  <td>{caps.default_steps}</td>
                  <td>
                    {caps.default_guidance ?? "—"}
                    {!caps.supports_guidance && <span className="hint"> fixed</span>}
                  </td>
                  <td>{caps.quantize ? `${caps.quantize} bits` : "—"}</td>
                  <td>
                    <div className="row" style={{ gap: 5 }}>
                      {caps.supports_negative_prompt && <span className="badge">negative</span>}
                      {caps.supports_image_to_image && <span className="badge">img2img</span>}
                      {caps.supports_edit && <span className="badge">editing</span>}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="hint" style={{ marginBottom: 0 }}>
            Start the server to see the catalogue and the declared capabilities.
          </p>
        )}
      </div>
    </>
  );
}
