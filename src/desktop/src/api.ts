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
  ModelStatus,
  Overview,
  Progress,
} from "./types";

// ── Rust commands ──────────────────────────────────────────────────────────

export const overview = () => invoke<Overview>("overview");
export const bootstrapRun = () => invoke<void>("bootstrap_run");
export const serverStart = () => invoke<number>("server_start");
export const serverStop = () => invoke<void>("server_stop");
export const serverRestart = () => invoke<number>("server_restart");
export const configRead = () => invoke<unknown>("config_read");
export const configWrite = (value: unknown) => invoke<void>("config_write", { value });
export const hfTokenWrite = (token: string) => invoke<void>("hf_token_write", { token });
export const prequantizeRun = (components: string[], dest?: string) =>
  invoke<void>("prequantize_run", { components, dest: dest ?? null });
export const modelsStatus = () => invoke<ModelStatus[]>("models_status");
export const modelFetch = (key: string) => invoke<void>("model_fetch", { key });

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
   * Subscribe to `/v1/progress`.
   *
   * `fetch` rather than `EventSource`: the latter allows no `Authorization`
   * header, and the route is protected like the rest of `/v1`. SSE framing is
   * simple enough to parse by hand.
   */
  subscribeProgress(
    onProgress: (progress: Progress) => void,
    onError: (message: string) => void,
  ): () => void {
    const controller = new AbortController();

    (async () => {
      try {
        const response = await fetch(this.url("/v1/progress"), {
          headers: { ...this.headers(), Accept: "text/event-stream" },
          signal: controller.signal,
        });
        if (!response.ok || !response.body) throw await describe(response);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
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
      } catch (error) {
        if (!controller.signal.aborted) onError(messageOf(error));
      }
    })();

    return () => controller.abort();
  }
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
