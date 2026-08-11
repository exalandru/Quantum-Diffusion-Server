import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";

import * as api from "./api";
import { ServerClient, messageOf } from "./api";
import { useJob } from "./job";
import { Configuration } from "./panels/Configuration";
import { Dashboard } from "./panels/Dashboard";
import { Logs } from "./panels/Logs";
import { Models } from "./panels/Models";
import { Setup } from "./panels/Setup";
import type { Overview, ServerLine } from "./types";

type View = "dashboard" | "models" | "config" | "logs";

const VIEWS: { id: View; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "models", label: "Models" },
  { id: "config", label: "Configuration" },
  { id: "logs", label: "Logs" },
];

/** Lines kept; past this point the view becomes unusable. */
const LOG_LIMIT = 2000;

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [state, setState] = useState<Overview | null>(null);
  const [config, setConfig] = useState<unknown>(null);
  //  Background-only. This is written by the pollers below and by nothing else:
  //  it used to be the single shared `error`, handed to every panel as `onError`
  //  while `refreshStatus` cleared it on every success — so an action's message
  //  survived at most four seconds. Actions own their own state (`actions.tsx`)
  //  and this channel reports one thing: whether the app can reach its backend.
  const [connectionError, setConnectionError] = useState<string | null>(null);
  //  Separate from the above for the same reason the two are separate from the
  //  panels: the config is read on mount and after a save, the status every four
  //  seconds. Sharing one slot would let the frequent poller's success erase the
  //  rare reader's failure — the original bug, one level down.
  const [configError, setConfigError] = useState<string | null>(null);
  const [lines, setLines] = useState<ServerLine[]>([]);
  // Above the views on purpose: a download or a conversion outlives the panel
  // that started it, so leaving Models and coming back must not lose sight of it.
  const jobs = useJob();

  // Status and configuration are polled separately on purpose. Re-reading the
  // config on a timer handed `Configuration` a brand-new object every tick, which
  // wiped whatever the user was typing — and rebuilt `client`, which tore down
  // and reopened the progress SSE stream every four seconds.
  const refreshStatus = useCallback(async () => {
    try {
      setState(await api.overview());
      // Clears only the background channel. Panels' action results are theirs.
      setConnectionError(null);
    } catch (cause) {
      setConnectionError(messageOf(cause));
    }
  }, []);

  const reloadConfig = useCallback(async () => {
    try {
      setConfig(await api.configRead());
      setConfigError(null);
    } catch (cause) {
      setConfigError(messageOf(cause));
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void reloadConfig();
    // The server process can die on its own: we re-poll the status rather than
    // trusting the last known state. The configuration, in contrast, only changes
    // when we change it.
    const timer = setInterval(() => void refreshStatus(), 4000);
    return () => clearInterval(timer);
  }, [refreshStatus, reloadConfig]);

  const onConfigChanged = useCallback(async () => {
    await Promise.all([reloadConfig(), refreshStatus()]);
  }, [reloadConfig, refreshStatus]);

  // Server output and conversion output arrive on the same channel.
  useEffect(() => {
    const pending = listen<ServerLine>("server-line", (event) => {
      setLines((previous) => {
        const next = [...previous, event.payload];
        return next.length > LOG_LIMIT ? next.slice(-LOG_LIMIT) : next;
      });
    });
    return () => {
      // Intentionally unhandled: this is teardown. A failed unsubscribe is not
      // actionable by the user, and the window is going away regardless.
      void pending.then((unlisten) => unlisten());
    };
  }, []);

  // Depend on the API key *string*, not on the config object. Even now that the
  // config is no longer polled, an object identity in this dependency list is
  // what silently reconnected the SSE stream — a primitive cannot regress that
  // way.
  const apiKey = useMemo(() => api.apiKeyOf(config), [config]);
  const client = useMemo(() => {
    const port = state?.server.port;
    if (!port || !state?.server.running) return null;
    return new ServerClient(port, apiKey);
  }, [state?.server.port, state?.server.running, apiKey]);

  // The background channel: what the app could not read on its own initiative.
  // Kept apart from the inline action notes, so a reader can tell "the app is
  // having trouble" from "the thing you just clicked failed".
  const background = (
    <>
      {connectionError && (
        <div className="notice notice-error" role="status">
          <strong>Background status check failed.</strong> {connectionError}
        </div>
      )}
      {configError && (
        <div className="notice notice-error" role="status">
          <strong>Could not read the configuration.</strong> {configError}
        </div>
      )}
    </>
  );

  if (!state) {
    return (
      <main>
        <header>
          <h1>Quantum Diffusion Server</h1>
        </header>
        {background}
        {!connectionError && <p className="empty">Loading…</p>}
      </main>
    );
  }

  // With no Python environment nothing else is actionable, so the installer is
  // shown alone rather than tabs that would all fail.
  if (!state.bootstrap.ready) {
    return (
      <main className="setup">
        {background}
        <Setup state={state} onDone={onConfigChanged} />
      </main>
    );
  }

  const running = state.server.running;

  return (
    <main className={view === "logs" ? "view-logs" : undefined}>
      <header>
        <h1>Quantum Diffusion Server</h1>
        {/* The server's own state, read from the status the app already polls.
            Not a second opinion about health: `overview` is the one source. */}
        <span className={running ? "pill pill-live" : "pill pill-down"}>
          {running ? "Running" : "Stopped"}
        </span>
        <nav className="views" role="tablist" aria-label="Views">
          {VIEWS.map(({ id, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={view === id}
              // A stable accessible name. Without it the Logs tab is announced as
              // "Logs 1"… "Logs 2" and renames itself several times a second while
              // the server talks, which is useless to a screen reader and makes
              // the control unaddressable by name.
              aria-label={label}
              className="view-tab"
              onClick={() => setView(id)}
            >
              {label}
              {id === "logs" && lines.length > 0 && (
                <span className="count" aria-hidden="true">
                  {lines.length}
                </span>
              )}
            </button>
          ))}
        </nav>
        {running && state.server.port !== null && (
          <code className="endpoint">127.0.0.1:{state.server.port}</code>
        )}
      </header>

      {background}

      {view === "dashboard" && (
        <Dashboard state={state} client={client} jobs={jobs} onChanged={refreshStatus} />
      )}
      {view === "models" && (
        <Models
          state={state}
          client={client}
          config={config}
          jobs={jobs}
          onConfigChanged={onConfigChanged}
        />
      )}
      {view === "config" && (
        <Configuration
          config={config}
          serverRunning={running}
          effectiveHfHome={state.hfHome}
          onSaved={onConfigChanged}
        />
      )}
      {view === "logs" && <Logs lines={lines} onClear={() => setLines([])} />}
    </main>
  );
}
