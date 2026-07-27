import { useEffect, useState } from "react";
import { open as openExternal } from "@tauri-apps/plugin-shell";

import * as api from "../api";
import { messageOf, type ServerClient } from "../api";
import type { Health, Overview, Progress } from "../types";

const IDLE: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

export function Dashboard({
  state,
  client,
  onChanged,
  onError,
}: {
  state: Overview;
  client: ServerClient | null;
  onChanged: () => void;
  onError: (message: string) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [progress, setProgress] = useState<Progress>(IDLE);

  // Progress arrives over Server-Sent Events, so without polling: we only
  // re-query `/health` for the warm model and idle memory.
  useEffect(() => {
    if (!client) {
      setHealth(null);
      setProgress(IDLE);
      return;
    }
    const stop = client.subscribeProgress(setProgress, () => {
      /* A dropped stream is not a visible error: the `overview` poll will notice
         a stopped server. */
    });
    void client.health().then(setHealth).catch(() => setHealth(null));
    return stop;
  }, [client]);

  useEffect(() => {
    if (!client) return;
    const timer = setInterval(() => {
      void client.health().then(setHealth).catch(() => undefined);
    }, 5000);
    return () => clearInterval(timer);
  }, [client]);

  async function act(label: string, action: () => Promise<unknown>) {
    setBusy(label);
    try {
      await action();
      onChanged();
    } catch (cause) {
      onError(messageOf(cause));
    } finally {
      setBusy(null);
    }
  }

  const running = state.server.running;
  const memory = progress.memory.active_gb !== undefined ? progress.memory : (health?.memory ?? {});
  const loaded = progress.loaded_model ?? health?.loaded_model ?? null;

  return (
    <>
      <div className="card">
        <div className="row spread">
          <div className="row">
            <h2 style={{ margin: 0 }}>Server</h2>
            <span className={`badge ${running ? "ok" : ""}`}>
              <span className="dot" />
              {running ? `running · port ${state.server.port}` : "stopped"}
            </span>
            {loaded ? (
              <span className="badge ok">warm model · {loaded}</span>
            ) : (
              running && <span className="badge">no model loaded</span>
            )}
          </div>
          <div className="row">
            {running ? (
              <>
                <button onClick={() => void act("stop", api.serverStop)} disabled={busy !== null}>
                  {busy === "stop" ? "Stopping…" : "Stop"}
                </button>
                <button
                  onClick={() => void act("restart", api.serverRestart)}
                  disabled={busy !== null}
                >
                  Restart
                </button>
              </>
            ) : (
              <button
                className="primary"
                onClick={() => void act("start", api.serverStart)}
                disabled={busy !== null}
              >
                {busy === "start" ? "Starting…" : "Start"}
              </button>
            )}
          </div>
        </div>

        {/* An unexpected exit is otherwise invisible: the status poll flips the
            badge back to "stopped" with nothing to say why. */}
        {!running && state.server.lastExit && (
          <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
            <span className="badge warn">
              <span className="dot" />
              {state.server.lastExit}
            </span>{" "}
            See the Logs tab for the reason.
          </p>
        )}

        <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
          The server listens within a second but loads no weights at startup: the first generation
          pays for loading the model, which can take several minutes.
        </p>

        {!state.hfTokenPresent && (
          <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
            <span className="badge warn">
              <span className="dot" />
              no HuggingFace token
            </span>{" "}
            The <code>black-forest-labs/*</code> repos are gated. Add a token in the Models tab,
            or the first download will fail.
          </p>
        )}
      </div>

      <div className="card">
        <h2>Activity</h2>
        {progress.state === "idle" ? (
          <p className="hint" style={{ marginBottom: 0 }}>
            {running ? "Idle." : "Server stopped."}
          </p>
        ) : (
          <>
            <div className="row spread" style={{ marginBottom: 8 }}>
              <span>
                {progress.state === "loading" ? (
                  <>
                    Loading <strong>{progress.model}</strong>…
                  </>
                ) : (
                  <>
                    <strong>{progress.model}</strong> · step {progress.step}/{progress.total}
                    {progress.seed !== null && <> · seed {progress.seed}</>}
                  </>
                )}
              </span>
              {progress.elapsed_s !== null && (
                <span className="badge">{progress.elapsed_s.toFixed(1)} s</span>
              )}
            </div>
            {/* Loading has no steps: indeterminate bar. */}
            {progress.state === "generating" && progress.total > 0 ? (
              <progress value={progress.step} max={progress.total} />
            ) : (
              <progress />
            )}
          </>
        )}

        <div className="row" style={{ marginTop: 12 }}>
          <button
            onClick={() => void act("cancel", () => client!.cancel())}
            disabled={!client || busy !== null || progress.state !== "generating"}
          >
            Cancel generation
          </button>
          <button
            onClick={() => void act("unload", () => client!.unload())}
            disabled={!client || busy !== null || !loaded}
          >
            {busy === "unload" ? "Releasing…" : "Free memory"}
          </button>
          <button onClick={() => void openExternal(client!.docsUrl())} disabled={!client}>
            Open /docs
          </button>
        </div>
        <p className="hint" style={{ marginTop: 8, marginBottom: 0 }}>
          MLX cannot be interrupted from outside: cancellation takes effect at the next denoising
          step.
        </p>
      </div>

      <div className="card">
        <h2>Memory and environment</h2>
        <dl className="stats">
          <div className="stat">
            <dt>MLX active</dt>
            <dd>{memory.active_gb !== undefined ? `${memory.active_gb.toFixed(2)} GB` : "—"}</dd>
          </div>
          <div className="stat">
            <dt>Peak</dt>
            <dd>{memory.peak_gb !== undefined ? `${memory.peak_gb.toFixed(2)} GB` : "—"}</dd>
          </div>
          <div className="stat">
            <dt>Cache</dt>
            <dd>{memory.cache_gb !== undefined ? `${memory.cache_gb.toFixed(2)} GB` : "—"}</dd>
          </div>
          <div className="stat">
            <dt>Server version</dt>
            <dd>{health?.version ?? "—"}</dd>
          </div>
        </dl>
        <p className="path" style={{ marginTop: 12 }}>
          {state.dataDir}
        </p>
      </div>
    </>
  );
}
