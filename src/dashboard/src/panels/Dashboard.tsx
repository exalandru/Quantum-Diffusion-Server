import { useEffect, useState } from "react";

import * as api from "../api";
import { messageOf } from "../api";
import { ActionNote, useActions } from "../actions";
import { describeJob, type JobView } from "../job";
import type { Health, Overview, Progress } from "../types";

const IDLE: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  preview_seq: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

/**
 * The operational home: what the server is doing, what it is doing it with, and
 * what it is using.
 *
 * Three groups, and the middle one is the reason the screen exists. Two very
 * different long things can be in flight — a generation, owned by the engine
 * and reported over SSE, and a download or conversion, owned by a child process
 * and reported by `/admin/jobs`. They have separate owners, separate
 * cancellation and separate failure modes, so they are shown as two things
 * rather than merged into a single invented "activity".
 */
export function Dashboard({
  state,
  health: polledHealth,
  jobs,
  onChanged,
}: {
  state: Overview;
  /** From the shell's own poll; this panel refines it, it does not own it. */
  health: Health | null;
  jobs: JobView;
  onChanged: () => void;
}) {
  const { run, dismiss, stateOf, busy, anyBusy } = useActions();
  const [health, setHealth] = useState<Health | null>(polledHealth);
  const [progress, setProgress] = useState<Progress>(IDLE);
  // Background readouts, reported where they are read rather than swallowed.
  // `/health` is a different question from `overview`: the supervisor can hold a
  // live child that is no longer serving, and `/v1/*` can 401 after an API-key
  // change while the process is perfectly healthy.
  const [healthNote, setHealthNote] = useState<string | null>(null);
  const [streamNote, setStreamNote] = useState<string | null>(null);

  // Progress arrives over Server-Sent Events, so without polling: we only
  // re-query `/health` for the warm model and idle memory.
  useEffect(() => {
    setStreamNote(null);
    const stop = api.subscribeProgress(
      (next) => {
        setProgress(next);
        setStreamNote(null);
      },
      // The transport retries on its own now; this only reports that it is
      // currently disconnected. A frame arriving clears it, and one arrives
      // immediately on a successful reconnect because `/v1/progress` emits the
      // current snapshot as its first event.
      (message) => setStreamNote(message),
    );
    return stop;
  }, []);

  // `/health` carries the warm model and the idle-release policy, which the
  // progress stream only reports while something is happening.
  useEffect(() => {
    const read = () =>
      void api
        .health()
        .then((value) => {
          setHealth(value);
          setHealthNote(null);
        })
        .catch((cause) => setHealthNote(messageOf(cause)));
    read();
    const timer = setInterval(read, 5000);
    return () => clearInterval(timer);
  }, []);

  // Each button gets its own slot, so a failed Stop cannot erase what Start said,
  // and nothing the status poll does can erase either.
  async function act(key: string, action: () => Promise<unknown>, success?: string) {
    if (await run(key, action, success)) onChanged();
  }

  const memory = progress.memory.active_gb !== undefined ? progress.memory : (health?.memory ?? {});
  const loaded = progress.loaded_model ?? health?.loaded_model ?? null;
  // "Free memory" releases both slots, so it must also be live when the only
  // thing resident is the upscaler.
  const holding = loaded !== null || (progress.upscaler ?? null) !== null;
  const job = jobs.job;
  const generating = progress.state !== "idle";
  const idleUnload = health?.idle_unload_s;

  return (
    <div className="panel">
      {/* One surface, three sections. The dashboard is read top to bottom — what
          the server is, what it is doing, what it is using — and three
          equal-height cards side by side made those look like three unrelated
          readings while padding the shortest out to match the tallest. A rule
          between sections separates them without the false symmetry. */}

      {/* ── Server ──────────────────────────────────────────────────────── */}
      <section className="section">
        <div className="row spread">
          <div className="row">
            <h2 style={{ margin: 0 }}>Server</h2>
            {/* No Running/Stopped pill and no Start button: this page is served
                by the server, so its existence answers the question, and
                nothing here has the authority to start one. That belongs to the
                menubar app, or to whoever typed `qds serve`. */}
            <span className="library-spec" style={{ marginLeft: 0 }}>
              {state.server.host}:{state.server.port}
            </span>
            {loaded ? (
              <span className="pill pill-ok">warm · {loaded}</span>
            ) : (
              <span className="pill">no model loaded</span>
            )}
            {/* Without this, an automatic release reads as a model that failed to
                stay loaded. */}
            {idleUnload !== null && idleUnload !== undefined && (
              <span className="pill">
                {idleUnload === 0 ? "frees after each request" : `frees after ${idleUnload}s idle`}
              </span>
            )}
          </div>

          {/* Restart survives because it is the one lifetime operation this page
              can honestly offer: the server re-execs itself, so the same process
              comes back and this page reconnects to it. Stopping from here would
              leave nothing to show the result on. */}
          <div className="actions">
            <button
              onClick={() =>
                void act("restart", api.serverRestart, "Restarting - this page will reconnect.")
              }
              disabled={anyBusy}
            >
              {busy("restart") ? "Restarting…" : "Restart server"}
            </button>
          </div>
        </div>

        {/* Failures stay until the same button runs again or they are dismissed
            — never cleared by a poll that happened to succeed. */}
        <ActionNote state={stateOf("restart")} onDismiss={() => dismiss("restart")} />

        <p className="note">
          The server listens within a second but loads no weights at startup: the first generation
          pays for loading the model, which can take several minutes.
        </p>

        {!state.hfTokenPresent && (
          <p className="note">
            <span className="pill pill-warn">no token</span> Half the catalogue sits in gated
            repositories. The models enabled by default are not among them; add a token under
            Configuration → Hugging Face before installing one that is.
          </p>
        )}
      </section>

      {/* ── Current activity ────────────────────────────────────────────── */}
      <section className="section">
        <h2>Current activity</h2>

        {/* Generation: owned by the Python engine, cancelled through it. */}
        {generating ? (
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
                <span className="library-spec">{progress.elapsed_s.toFixed(1)} s</span>
              )}
            </div>
            {/* Loading reports no steps, so it gets a stripe rather than a
                fraction it cannot know. */}
            {progress.state === "generating" && progress.total > 0 ? (
              <div className="bar">
                <div
                  className="bar-fill"
                  style={{ width: `${(progress.step / progress.total) * 100}%` }}
                />
              </div>
            ) : (
              <div className="bar bar-indeterminate" />
            )}
          </>
        ) : (
          <p className="empty">No generation running.</p>
        )}

        {/* The job, which is a separate owner from the generation above: a
            download or a conversion is a child process with its own
            cancellation and its own failure modes. */}
        {job && jobs.active && (
          <div style={{ marginTop: 14 }}>
            <div className="row" style={{ marginBottom: 6 }}>
              <span className="pill pill-accent">
                {job.kind === "prequantize" ? "Converting" : "Downloading"}
              </span>
              {job.target && <strong>{job.target}</strong>}
              {job.state === "cancelling" && <span className="pill pill-warn">stopping</span>}
            </div>
            <p className="note" style={{ marginTop: 0, marginBottom: 8 }}>
              {describeJob(job)}
            </p>
            <div className="bar bar-indeterminate" />
            <p className="note">Manage it in Models, where it was started.</p>
          </div>
        )}

        <div className="actions" style={{ marginTop: 14 }}>
          <button
            onClick={() => void act("cancel", api.cancelGeneration, "Cancellation requested.")}
            // An upscale is cancellable too — it stops at the next tile — so
            // this must not stay inert while one runs.
            disabled={anyBusy || (progress.state !== "generating" && progress.state !== "upscaling")}
            title={
              progress.state === "generating" || progress.state === "upscaling"
                ? undefined
                : "Available while a generation is running."
            }
          >
            {busy("cancel")
              ? "Cancelling…"
              : progress.state === "upscaling"
                ? "Cancel upscale"
                : "Cancel generation"}
          </button>
          <button
            onClick={() => void act("unload", api.unload, "Model released.")}
            disabled={anyBusy || !holding}
            title={holding ? undefined : "Nothing is loaded."}
          >
            {busy("unload") ? "Releasing…" : "Free memory"}
          </button>
          {/* A plain link now that this is a web page: `/docs` is served by the
              same origin, so there is nothing to ask a native shell to open. */}
          <a className="button" href={api.docsUrl()} target="_blank" rel="noreferrer">
            Open /docs
          </a>
        </div>
        {["cancel", "unload"].map((key) => (
          <ActionNote key={key} state={stateOf(key)} onDismiss={() => dismiss(key)} />
        ))}

        {/* Shown only while actually disconnected: the transport retries with a
            bounded backoff and the first frame of a recovered stream clears this.
            A silently stopped bar would be the lie; so would telling someone to
            go and fix it by hand. */}
        {streamNote && (
          <p className="note">
            <span className="pill pill-warn">reconnecting</span> Live progress stopped updating:{" "}
            {streamNote}. Retrying automatically.
          </p>
        )}
        <p className="note">
          MLX cannot be interrupted from outside: cancellation takes effect at the next denoising
          step.
        </p>
      </section>

      {/* ── Runtime ─────────────────────────────────────────────────────── */}
      <section className="section">
        <h2>Runtime</h2>
        <dl className="stats">
          <div className="stat">
            <dt>Warm model</dt>
            <dd>{loaded ?? "none"}</dd>
          </div>
          <div className="stat">
            <dt>MLX active</dt>
            <dd>{memory.active_gb !== undefined ? `${memory.active_gb.toFixed(2)} GB` : "-"}</dd>
          </div>
          <div className="stat">
            <dt>Peak</dt>
            <dd>{memory.peak_gb !== undefined ? `${memory.peak_gb.toFixed(2)} GB` : "-"}</dd>
          </div>
          <div className="stat">
            <dt>Cache</dt>
            <dd>{memory.cache_gb !== undefined ? `${memory.cache_gb.toFixed(2)} GB` : "-"}</dd>
          </div>
          <div className="stat">
            <dt>Server version</dt>
            <dd>{health?.version ?? "-"}</dd>
          </div>
          <div className="stat">
            <dt>Release policy</dt>
            {/* Three states, not two. `null` from a *running* server means "keep
                the model warm forever"; no reading at all means we do not know,
                and printing "keep warm" for it asserted the opposite of a
                configured 10-second release — observed with the server stopped. */}
            <dd>
              {!health
                ? "-"
                : idleUnload === null || idleUnload === undefined
                  ? "keep warm"
                  : idleUnload === 0
                    ? "after each request"
                    : `after ${idleUnload}s idle`}
            </dd>
          </div>
          <div className="stat wide">
            <dt>Working directory</dt>
            <dd>{state.dataDir}</dd>
          </div>
          <div className="stat wide">
            <dt>Model storage</dt>
            <dd>{state.effectiveHfHome}</dd>
          </div>
        </dl>
        {healthNote && (
          <p className="note">
            <span className="pill pill-warn">/health unreachable</span> {healthNote}
          </p>
        )}
        {jobs.error && (
          <p className="note">
            <span className="pill pill-warn">operation status unreadable</span> {jobs.error}
          </p>
        )}
      </section>
    </div>
  );
}
