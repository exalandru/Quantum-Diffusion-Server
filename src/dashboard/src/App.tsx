import { useCallback, useEffect, useState } from "react";

import * as api from "./api";
import { Unauthorized, messageOf } from "./api";
import { useJob } from "./job";
import { useLogs } from "./logs";
import { LoginPrompt } from "./LoginPrompt";
import { Configuration } from "./panels/Configuration";
import { Dashboard } from "./panels/Dashboard";
import { Logs } from "./panels/Logs";
import { Models } from "./panels/Models";
import { Unreachable } from "./Unreachable";
import type { ConfigDocument, Health, Overview } from "./types";

type View = "dashboard" | "models" | "config" | "logs";

const VIEWS: { id: View; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "models", label: "Models" },
  { id: "config", label: "Configuration" },
  { id: "logs", label: "Logs" },
];

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [state, setState] = useState<Overview | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [config, setConfig] = useState<ConfigDocument | null>(null);
  //  Background-only. This is written by the pollers below and by nothing else:
  //  it used to be the single shared `error`, handed to every panel as `onError`
  //  while `refreshStatus` cleared it on every success — so an action's message
  //  survived at most four seconds. Actions own their own state (`actions.tsx`)
  //  and this channel reports one thing: whether the page can reach its server.
  const [connectionError, setConnectionError] = useState<string | null>(null);
  //  Separate from the above for the same reason the two are separate from the
  //  panels: the config is read on mount and after a save, the status every four
  //  seconds. Sharing one slot would let the frequent poller's success erase the
  //  rare reader's failure — the original bug, one level down.
  const [configError, setConfigError] = useState<string | null>(null);
  //  Demanded, not merely absent: shown after the server answers 401, so a
  //  server with no password set never puts a login screen in the way.
  const [session, setSession] = useState<api.SessionStatus | null>(null);
  // Above the views on purpose: a download or a conversion outlives the panel
  // that started it, so leaving Models and coming back must not lose sight of it.
  const jobs = useJob();
  const logs = useLogs();

  const onUnauthorized = useCallback(() => {
    // Ask the server which of the two situations this is — no password set, or
    // one set and not presented — rather than guessing from the status code.
    void api.sessionStatus().then(setSession).catch(() => setSession(null));
  }, []);

  // Status and configuration are polled separately on purpose. Re-reading the
  // config on a timer handed `Configuration` a brand-new object every tick,
  // which wiped whatever the user was typing.
  const refreshStatus = useCallback(async () => {
    try {
      const [next, nextHealth] = await Promise.all([api.overview(), api.health()]);
      setState(next);
      setHealth(nextHealth);
      setSession(null);
      // Clears only the background channel. Panels' action results are theirs.
      setConnectionError(null);
    } catch (cause) {
      if (cause instanceof Unauthorized) {
        onUnauthorized();
        return;
      }
      setConnectionError(messageOf(cause));
    }
  }, [onUnauthorized]);

  const reloadConfig = useCallback(async () => {
    try {
      setConfig(await api.configRead());
      setConfigError(null);
    } catch (cause) {
      if (cause instanceof Unauthorized) return onUnauthorized();
      setConfigError(messageOf(cause));
    }
  }, [onUnauthorized]);

  useEffect(() => {
    void refreshStatus();
    void reloadConfig();
    // The server can go away under us — it is restarted from this very page,
    // and the menubar app can stop it — so the status is re-polled rather than
    // trusted. The configuration, in contrast, only changes when we change it.
    const timer = setInterval(() => void refreshStatus(), 4000);
    return () => clearInterval(timer);
  }, [refreshStatus, reloadConfig]);

  const onConfigChanged = useCallback(async () => {
    await Promise.all([reloadConfig(), refreshStatus()]);
  }, [reloadConfig, refreshStatus]);

  if (session) {
    return (
      <LoginPrompt
        status={session}
        onAuthenticated={() => {
          setSession(null);
          void refreshStatus();
          void reloadConfig();
        }}
      />
    );
  }

  // Not an error banner over an empty shell: with no answer at all there is
  // nothing to render tabs *about*, and the one useful thing to say is how to
  // start the server. `refreshStatus` keeps polling behind this, so it clears
  // itself the moment the server comes back.
  if (!state && connectionError) {
    return <Unreachable reason={connectionError} />;
  }

  if (!state) {
    return (
      <main>
        <header>
          <h1>Quantum Diffusion Server</h1>
        </header>
        <p className="empty">Loading…</p>
      </main>
    );
  }

  // The background channel: what the page could not read on its own initiative.
  // Kept apart from the inline action notes, so a reader can tell "the page is
  // having trouble" from "the thing you just clicked failed".
  const background = (
    <>
      {state.recoveryMode && (
        <div className="notice notice-error" role="status">
          <strong>The server started in recovery mode.</strong> {state.recoveryError} Generation is
          unavailable until the configuration below loads; fix it and restart.
        </div>
      )}
      {state.restartRequired && !state.recoveryMode && (
        <div className="notice" role="status">
          <strong>A restart is needed</strong> for the saved configuration to take effect.
        </div>
      )}
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

  // In recovery mode the server has no models and no engine, so every view but
  // the one that repairs the configuration would render an error.
  const available = state.recoveryMode ? VIEWS.filter((entry) => entry.id === "config") : VIEWS;
  const current = state.recoveryMode ? "config" : view;

  return (
    <main className={current === "logs" ? "view-logs" : undefined}>
      {/* Two rows, one `header`. Two rows because the tabs used to sit on the
          same line as the title, the status pill and the endpoint, at the same
          weight — so the four things you navigate with read as more of the
          masthead. One element because `main.view-logs > header` is a
          direct-child selector (styles.css) and `Logs.layout.test.ts` renders
          exactly that chain: splitting this into siblings silently returns the
          Logs pane to a fixed box. */}
      <header>
        <div className="identity">
          <h1>Quantum Diffusion Server</h1>
          <span className={state.recoveryMode ? "pill pill-down" : "pill pill-live"}>
            {state.recoveryMode ? "Recovery" : "Running"}
          </span>
          <code className="endpoint">
            {state.server.host}:{state.server.port}
          </code>
        </div>
        <nav className="views" role="tablist" aria-label="Views">
          {available.map(({ id, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={current === id}
              // A stable accessible name. Without it the Logs tab is announced as
              // "Logs 1"… "Logs 2" and renames itself several times a second while
              // the server talks, which is useless to a screen reader and makes
              // the control unaddressable by name.
              aria-label={label}
              className="view-tab"
              onClick={() => setView(id)}
            >
              {label}
              {id === "logs" && logs.entries.length > 0 && (
                <span className="count" aria-hidden="true">
                  {logs.entries.length}
                </span>
              )}
            </button>
          ))}
        </nav>
      </header>

      {background}

      {current === "dashboard" && (
        <Dashboard state={state} health={health} jobs={jobs} onChanged={refreshStatus} />
      )}
      {current === "models" && (
        <Models state={state} config={config} jobs={jobs} onConfigChanged={onConfigChanged} />
      )}
      {current === "config" && (
        <Configuration
          config={config}
          effectiveHfHome={state.effectiveHfHome}
          defaultCacheDir={state.effectiveCacheDir}
          hfTokenPresent={state.hfTokenPresent}
          adminPasswordSet={state.adminPasswordSet}
          lanAddresses={state.lanAddresses}
          onSaved={onConfigChanged}
        />
      )}
      {current === "logs" && <Logs entries={logs.entries} dropped={logs.dropped} onClear={logs.clear} />}
    </main>
  );
}
