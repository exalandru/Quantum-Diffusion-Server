// First run, update and repair of the managed Python runtime.
//
// Shown alone, without the views behind it: with no runtime, every other control
// would fail, and offering them would only produce errors the user cannot act on.
//
// The install is minutes of downloading, so it never happens as a side effect of
// pressing anything else. uv's own output is relayed rather than replaced by an
// indeterminate spinner: a progress bar that cannot report progress is a lie
// about how long this takes.
//
// Every word below is chosen by the backend's `bootstrap.state` (Slice 8).
// React reconstructs nothing: "installed but interrupted" and "never installed"
// are different states on disk, and the screen must not collapse them the way
// the old `installedVersion !== null` guess did.

import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";

import * as api from "../api";
import { messageOf } from "../api";
import type { BootstrapEvent, BootstrapState, Overview } from "../types";

export function Setup({ state, onDone }: { state: Overview; onDone: () => void }) {
  const bootstrap = state.bootstrap;
  const [running, setRunning] = useState(bootstrap.state === "installing");
  const [step, setStep] = useState<string | null>(null);
  const [output, setOutput] = useState<string[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const log = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const pending = listen<BootstrapEvent>("bootstrap", (event) => {
      const payload = event.payload;
      switch (payload.kind) {
        case "step":
          setStep(payload.message);
          break;
        case "output":
          setOutput((lines) => [...lines.slice(-400), payload.line]);
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
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [output]);

  async function initialize() {
    setRunning(true);
    setFailure(null);
    setOutput([]);
    try {
      const started = await api.bootstrapRun();
      if (!started) {
        // Rust holds the single-flight guard, so this is authoritative: another
        // window, or a click this one never saw, got there first.
        setFailure("An installation is already running. Its progress appears below.");
      }
    } catch (cause) {
      setRunning(false);
      setFailure(messageOf(cause));
    }
  }

  const { title, explanation, action } = describe(bootstrap.state, bootstrap);
  const busy = running || bootstrap.state === "installing";

  return (
    <section className="panel">
      <h2>{title}</h2>
      <p className="prose">{explanation}</p>

      <dl className="setup-facts">
        <dt>Runtime</dt>
        <dd className="wrap">{bootstrap.envPath}</dd>
        <dt>Working directory</dt>
        <dd className="wrap">{state.dataDir}</dd>
        <dt>Application</dt>
        <dd>{bootstrap.appVersion}</dd>
        {bootstrap.installedVersion && (
          <>
            <dt>Installed by</dt>
            <dd>{bootstrap.installedVersion}</dd>
          </>
        )}
      </dl>

      <div className="actions">
        <button className="primary" onClick={() => void initialize()} disabled={busy}>
          {busy ? "Working…" : action}
        </button>
        {step && <span className="pill pill-live">{step}</span>}
      </div>

      {/* What the *previous* attempt reported, from the durable record — distinct
          from `failure`, which is what this session's attempt just said. */}
      {bootstrap.failure && !running && (
        <div className="notice notice-warn" style={{ marginTop: 14, marginBottom: 0 }}>
          <strong>The last attempt stopped:</strong> {bootstrap.failure}
        </div>
      )}

      {failure && (
        <div className="notice notice-error" role="alert" style={{ marginTop: 14, marginBottom: 0 }}>
          {failure}
        </div>
      )}

      {output.length > 0 && (
        <pre className="setup-log" ref={log}>
          {output.join("\n")}
        </pre>
      )}
    </section>
  );
}

/** What to say for each backend state, and what the button does. */
function describe(
  state: BootstrapState,
  bootstrap: Overview["bootstrap"],
): { title: string; explanation: string; action: string } {
  const sameVersion = bootstrap.installedVersion === bootstrap.appVersion;
  switch (state) {
    case "installing":
      return {
        title: "Setting up",
        explanation: "Installing Python and the locked dependencies.",
        action: "Working…",
      };
    case "updateRequired":
      return {
        title: "Update the environment",
        // Two different facts, and saying "version X while the app runs Y" with
        // X === Y would read as a bug. The payload fingerprint is what changed.
        explanation: sameVersion
          ? "The bundled server has changed since the environment was installed, so the installed " +
            "copy would keep answering with the old code. Rebuilding reinstalls it — the " +
            "dependencies are already downloaded, so this is usually quick, and it touches " +
            "neither your models nor your settings."
          : `The environment installed here was built by version ${bootstrap.installedVersion}, ` +
            `while the app runs ${bootstrap.appVersion}. It has to be rebuilt. Your models and ` +
            "settings are stored separately and are left untouched.",
        action: "Rebuild",
      };
    case "broken":
      return {
        title: "Repair the environment",
        explanation:
          "An installation was started here and never finished, so the environment cannot be " +
          "trusted to hold the code this app expects. Repairing recopies the server and " +
          "reinstalls the dependencies. Nothing you have downloaded or imported is touched.",
        action: "Repair",
      };
    default:
      return {
        title: "Install the environment",
        explanation:
          "The app installs its own Python and dependencies, so it needs nothing from your " +
          "machine — no system Python, no Homebrew, no terminal setup. Expect about 1.1 GB of " +
          "downloading and a few minutes. Model weights are separate and come later, on demand.",
        action: "Install",
      };
  }
}
