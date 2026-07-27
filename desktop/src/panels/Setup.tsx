import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";

import * as api from "../api";
import { messageOf } from "../api";
import type { BootstrapEvent, Overview } from "../types";

/**
 * First launch: installing the Python environment.
 *
 * This is roughly a 1.1 GB download — torch alone is 501 MB, and mlx 178 of
 * which 150 are Metal shaders. Hence uv's live output rather than an
 * indeterminate spinner.
 */
export function Setup({ state, onDone }: { state: Overview; onDone: () => void }) {
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [output, setOutput] = useState<string[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const console_ = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const pending = listen<BootstrapEvent>("bootstrap", (event) => {
      const payload = event.payload;
      switch (payload.kind) {
        case "step":
          setStep(payload.message);
          break;
        case "output":
          setOutput((previous) => [...previous.slice(-400), payload.line]);
          break;
        case "done":
          setRunning(false);
          setStep(null);
          onDone();
          break;
        case "failed":
          setRunning(false);
          setFailure(payload.message);
          break;
      }
    });
    return () => {
      void pending.then((unlisten) => unlisten());
    };
  }, [onDone]);

  useEffect(() => {
    console_.current?.scrollTo({ top: console_.current.scrollHeight });
  }, [output]);

  async function install() {
    setRunning(true);
    setFailure(null);
    setOutput([]);
    try {
      await api.bootstrapRun();
    } catch (cause) {
      setRunning(false);
      setFailure(messageOf(cause));
    }
  }

  const upgrade = state.bootstrap.installedVersion !== null;

  return (
    <div className="card">
      <h2>{upgrade ? "Updating the environment" : "Installation"}</h2>
      <p className="hint">
        {upgrade ? (
          <>
            The environment currently installed was built by version{" "}
            <strong>{state.bootstrap.installedVersion}</strong>, while the app runs{" "}
            <strong>{state.bootstrap.appVersion}</strong>. It has to be rebuilt.
          </>
        ) : (
          <>
            The app installs its own Python and dependencies, requiring nothing from the machine.
            Expect about <strong>1.1 GB</strong> of download and a few minutes. The model weights
            themselves come later, on demand.
          </>
        )}
      </p>

      <dl className="stats">
        <div className="stat">
          <dt>Destination</dt>
          <dd className="path">{state.bootstrap.envPath}</dd>
        </div>
        <div className="stat">
          <dt>Working directory</dt>
          <dd className="path">{state.dataDir}</dd>
        </div>
      </dl>

      <div className="row" style={{ marginTop: 14 }}>
        <button className="primary" onClick={install} disabled={running}>
          {running ? "Installing…" : upgrade ? "Rebuild" : "Install"}
        </button>
        {step && <span className="badge">{step}</span>}
      </div>

      {failure && (
        <div className="error-banner" style={{ marginTop: 14 }}>
          {failure}
        </div>
      )}

      {output.length > 0 && (
        <div className="console" ref={console_} style={{ marginTop: 14 }}>
          {output.map((line, index) => (
            <div key={index}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}
