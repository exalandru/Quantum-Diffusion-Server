import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";

import * as api from "../api";
import { messageOf } from "../api";
import type { BootstrapEvent, Overview } from "../types";

/**
 * Premier lancement : installation de l'environnement Python.
 *
 * C'est un téléchargement d'environ 1,1 Go — torch pèse 501 Mo à lui seul, et
 * mlx 178 dont 150 de shaders Metal. D'où la sortie d'uv en direct plutôt qu'un
 * indicateur d'attente indéterminé.
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
      <h2>{upgrade ? "Mise à jour de l'environnement" : "Installation"}</h2>
      <p className="hint">
        {upgrade ? (
          <>
            L'environnement présent a été installé par la version{" "}
            <strong>{state.bootstrap.installedVersion}</strong>, l'app tourne en{" "}
            <strong>{state.bootstrap.appVersion}</strong>. Il faut le reconstruire.
          </>
        ) : (
          <>
            L'app installe son propre Python et ses dépendances, sans rien exiger de la machine.
            Compter environ <strong>1,1 Go</strong> de téléchargement et quelques minutes. Les poids
            des modèles, eux, viendront plus tard et à la demande.
          </>
        )}
      </p>

      <dl className="stats">
        <div className="stat">
          <dt>Destination</dt>
          <dd className="path">{state.bootstrap.envPath}</dd>
        </div>
        <div className="stat">
          <dt>Espace de travail</dt>
          <dd className="path">{state.dataDir}</dd>
        </div>
      </dl>

      <div className="row" style={{ marginTop: 14 }}>
        <button className="primary" onClick={install} disabled={running}>
          {running ? "Installation…" : upgrade ? "Reconstruire" : "Installer"}
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
