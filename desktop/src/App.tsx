import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";

import * as api from "./api";
import { ServerClient, messageOf } from "./api";
import { Configuration } from "./panels/Configuration";
import { Dashboard } from "./panels/Dashboard";
import { Logs } from "./panels/Logs";
import { Models } from "./panels/Models";
import { Setup } from "./panels/Setup";
import type { Overview, ServerLine } from "./types";

type Tab = "dashboard" | "config" | "models" | "logs";

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "config", label: "Configuration" },
  { id: "models", label: "Models" },
  { id: "logs", label: "Logs" },
];

/** Lines kept; past this point the view becomes unusable. */
const LOG_LIMIT = 2000;

export function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [state, setState] = useState<Overview | null>(null);
  const [config, setConfig] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState<ServerLine[]>([]);

  const refresh = useCallback(async () => {
    try {
      const [next, configuration] = await Promise.all([api.overview(), api.configRead()]);
      setState(next);
      setConfig(configuration);
      setError(null);
    } catch (cause) {
      setError(messageOf(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
    // The server process can die on its own: we re-poll regularly rather than
    // trusting the last known state.
    const timer = setInterval(() => void refresh(), 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  // Server output and conversion output arrive on the same channel.
  useEffect(() => {
    const pending = listen<ServerLine>("server-line", (event) => {
      setLines((previous) => {
        const next = [...previous, event.payload];
        return next.length > LOG_LIMIT ? next.slice(-LOG_LIMIT) : next;
      });
    });
    return () => {
      void pending.then((unlisten) => unlisten());
    };
  }, []);

  const client = useMemo(() => {
    const port = state?.server.port;
    if (!port || !state?.server.running) return null;
    return new ServerClient(port, api.apiKeyOf(config));
  }, [state?.server.port, state?.server.running, config]);

  if (!state) {
    return (
      <div className="shell">
        <div className="titlebar">Quantum Diffusion Server</div>
        <main>
          {error ? <div className="error-banner">{error}</div> : <p className="center-note">Loading…</p>}
        </main>
      </div>
    );
  }

  // With no Python environment nothing else is actionable, so we show the
  // installer alone rather than tabs that would all fail.
  if (!state.bootstrap.ready) {
    return (
      <div className="shell">
        <div className="titlebar">Quantum Diffusion Server</div>
        <main>
          {error && <div className="error-banner">{error}</div>}
          <Setup state={state} onDone={refresh} />
        </main>
      </div>
    );
  }

  return (
    <div className="shell">
      <div className="titlebar">Quantum Diffusion Server</div>
      <div className="tabs" role="tablist">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            role="tab"
            className="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
            {id === "logs" && lines.length > 0 && <span className="badge">{lines.length}</span>}
          </button>
        ))}
      </div>
      <main>
        {error && <div className="error-banner">{error}</div>}
        {tab === "dashboard" && (
          <Dashboard state={state} client={client} onChanged={refresh} onError={setError} />
        )}
        {tab === "config" && (
          <Configuration
            config={config}
            client={client}
            serverRunning={state.server.running}
            onSaved={refresh}
            onError={setError}
          />
        )}
        {tab === "models" && (
          <Models state={state} client={client} config={config} onError={setError} />
        )}
        {tab === "logs" && <Logs lines={lines} onClear={() => setLines([])} />}
      </main>
    </div>
  );
}
