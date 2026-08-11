/**
 * Two channels, deliberately distinct.
 *
 * The Rust side handles only what the browser cannot: installing the
 * environment, supervising a process, writing a file. Everything else goes
 * through the server's HTTP API, queried directly — funnelling it across the IPC
 * bridge would only add a layer to debug.
 */
import { invoke } from "@tauri-apps/api/core";

import type {
  Capabilities,
  Health,
  ImportVerdict,
  LocateVerdict,
  JobStatus,
  ModelStatus,
  Overview,
  Progress,
} from "./types";

// ── Rust commands ──────────────────────────────────────────────────────────

export const overview = () => invoke<Overview>("overview");
/** `false` when an install was already running, so this call started nothing. */
export const bootstrapRun = () => invoke<boolean>("bootstrap_run");
export const serverStart = () => invoke<number>("server_start");
export const serverStop = () => invoke<void>("server_stop");
export const serverRestart = () => invoke<number>("server_restart");
export const configRead = () => invoke<unknown>("config_read");
export const configWrite = (value: unknown) => invoke<void>("config_write", { value });
export const hfTokenWrite = (token: string) => invoke<void>("hf_token_write", { token });
export const prequantizeRun = (
  model: string,
  bits: number,
  components: string[] = [],
  dest?: string,
) => invoke<void>("prequantize_run", { model, bits, components, dest: dest ?? null });
export const modelsStatus = () => invoke<ModelStatus[]>("models_status");
/** Starts the download and returns once the child is running, not once it is done. */
export const modelFetch = (key: string) => invoke<void>("model_fetch", { key });
export const jobStatus = () => invoke<JobStatus>("job_status");
export const jobCancel = () => invoke<JobStatus>("job_cancel");
/** Native macOS folder chooser. `null` when the user cancels. */
export const pickDirectory = (start?: string | null) =>
  invoke<string | null>("pick_directory", { start: start ?? null });
export const localModelInspect = (path: string) =>
  invoke<ImportVerdict>("local_model_inspect", { path });
export const localModelRegister = (
  path: string,
  baseProfile: string,
  name?: string,
  apiName?: string,
) =>
  invoke<{ ok: boolean; reason?: string; already_imported?: boolean }>("local_model_register", {
    path,
    baseProfile,
    name: name ?? null,
    apiName: apiName ?? null,
  });
/** Check a directory against a built-in entry. Binds nothing. */
export const localModelLocate = (path: string, model: string) =>
  invoke<LocateVerdict>("local_model_locate", { path, model });
export const localModelForget = (id: string) =>
  invoke<{ ok: boolean; reason?: string }>("local_model_forget", { id });

// ── Server HTTP client ─────────────────────────────────────────────────────

export class ServerClient {
  constructor(
    private readonly port: number,
    private readonly apiKey: string | null,
  ) {}

  private headers(): HeadersInit {
    return this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {};
  }

  private url(path: string): string {
    return `http://127.0.0.1:${this.port}${path}`;
  }

  private async get<T>(path: string): Promise<T> {
    const response = await fetch(this.url(path), { headers: this.headers() });
    if (!response.ok) throw await describe(response);
    return (await response.json()) as T;
  }

  private async post<T>(path: string): Promise<T> {
    const response = await fetch(this.url(path), { method: "POST", headers: this.headers() });
    if (!response.ok) throw await describe(response);
    return (await response.json()) as T;
  }

  /** Unauthenticated: a probe available even with a key configured. */
  health = () => this.get<Health>("/health");
  capabilities = () => this.get<Capabilities>("/v1/capabilities");
  cancel = () => this.post<{ cancelled: boolean; state: string }>("/v1/cancel");
  unload = () => this.post<{ loaded_model: string | null; memory: Health["memory"] }>("/v1/unload");

  docsUrl = () => this.url("/docs");

  /**
   * Subscribe to `/v1/progress`, reconnecting until told to stop.
   *
   * `fetch` rather than `EventSource`: the latter allows no `Authorization`
   * header, and the route is protected like the rest of `/v1`. SSE framing is
   * simple enough to parse by hand — and doing so is what makes the retry below
   * possible, since `EventSource`'s own reconnect is neither bounded nor
   * cancellable on our terms.
   *
   * Retrying here rather than in the panel is deliberate. One loop owns the
   * connection, so "only one stream at a time" is structural rather than a rule
   * someone has to remember: each attempt is awaited before the next begins, and
   * the caller's single unsubscribe cancels both the in-flight request and any
   * pending wait.
   *
   * Two different endings are both drops. A thrown error is the obvious one; a
   * *clean* end — the server closing the response — used to fall out of the read
   * loop and return silently, leaving a panel subscribed to nothing at all with
   * no indication. Both now schedule a retry.
   */
  subscribeProgress(
    onProgress: (progress: Progress) => void,
    onError: (message: string) => void,
  ): () => void {
    let stopped = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    /** Resolves the pending backoff early, so unsubscribing is immediate. */
    let wake: (() => void) | null = null;
    let attempt = 0;

    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        wake = resolve;
        timer = setTimeout(resolve, ms);
      });

    const consume = async (signal: AbortSignal) => {
      const response = await fetch(this.url("/v1/progress"), {
        headers: { ...this.headers(), Accept: "text/event-stream" },
        signal,
      });
      if (!response.ok || !response.body) throw await describe(response);
      // The connection opened, so whatever went wrong before is over: the next
      // failure starts its backoff from the beginning rather than inheriting a
      // ten-second delay from an outage that has since been fixed.
      attempt = 0;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line.
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const payload = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("");
          // Keep-alive `: ping` frames have no `data:`: nothing to do.
          if (payload) onProgress(JSON.parse(payload) as Progress);
          boundary = buffer.indexOf("\n\n");
        }
      }
    };

    void (async () => {
      while (!stopped) {
        const attemptController = new AbortController();
        controller = attemptController;
        try {
          await consume(attemptController.signal);
          if (stopped) return;
          onError("the server closed the progress stream");
        } catch (error) {
          // An abort is us, not a fault: unsubscribing must not report an error
          // to a panel that is on its way out.
          if (stopped || attemptController.signal.aborted) return;
          onError(messageOf(error));
        } finally {
          // Whatever ended this attempt, the request that carried it is finished
          // with. Without this a failure that is *not* the connection closing —
          // a malformed frame, say — would leave the previous body draining
          // while the retry opens a second one, which is the duplicate-stream
          // case this design exists to make impossible.
          attemptController.abort();
        }
        if (stopped) return;
        await sleep(backoffMs(attempt++));
      }
    })();

    return () => {
      stopped = true;
      controller?.abort();
      if (timer) clearTimeout(timer);
      // Let the loop observe `stopped` and finish, rather than leaving its
      // promise suspended for the lifetime of the page.
      wake?.();
    };
  }
}

/** First retry, and the ceiling. Doubling in between. */
export const RECONNECT_BASE_MS = 500;
export const RECONNECT_MAX_MS = 10_000;

/**
 * How long to wait before retry number `attempt` (0-based).
 *
 * Bounded on purpose: a local server that is genuinely down should be retried
 * every ten seconds forever — often enough that restarting it recovers the view
 * without touching anything, rarely enough that a stopped server does not
 * produce a request storm in the log for as long as the app is open.
 */
export function backoffMs(attempt: number): number {
  return Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** Math.max(0, attempt));
}

async function describe(response: Response): Promise<Error> {
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    if (body.error?.message) return new Error(body.error.message);
  } catch {
    // Non-JSON body: fall back to the status.
  }
  return new Error(`HTTP ${response.status}`);
}

export function messageOf(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return JSON.stringify(error);
}

/** Pull the API key out of the configuration, to authenticate calls. */
export function apiKeyOf(config: unknown): string | null {
  const server = (config as { server?: { api_key?: unknown } } | null)?.server;
  const key = server?.api_key;
  return typeof key === "string" && key.length > 0 ? key : null;
}
