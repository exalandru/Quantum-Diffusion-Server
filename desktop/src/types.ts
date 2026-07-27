/** Types shared with the Rust side and with the server's HTTP API. */

// ── Rust side ──────────────────────────────────────────────────────────────

export type ServerStatus = {
  running: boolean;
  port: number | null;
  lastExit: string | null;
};

export type BootstrapStatus = {
  ready: boolean;
  installedVersion: string | null;
  appVersion: string;
  envPath: string;
};

export type Overview = {
  server: ServerStatus;
  bootstrap: BootstrapStatus;
  hfTokenPresent: boolean;
  hfHome: string;
  dataDir: string;
  configPath: string;
  flux2DevReady: boolean;
};

export type BootstrapEvent =
  | { kind: "step"; message: string }
  | { kind: "output"; line: string }
  | { kind: "done" }
  | { kind: "failed"; message: string };

/** One line of server output. `structured` ⇒ stdout, hence JSON. */
export type ServerLine = { structured: boolean; line: string };

// ── HTTP server side ───────────────────────────────────────────────────────

export type Health = {
  status: string;
  version: string;
  default_model: string;
  models: string[];
  loaded_model: string | null;
  /** Seconds of inactivity before the model is released; `null` = never. */
  idle_unload_s: number | null;
  /** Empty when `mlx.core` is unavailable: treat the keys as optional. */
  memory: { active_gb?: number; peak_gb?: number; cache_gb?: number };
};

export type ModelCapabilities = {
  repo: string;
  default_size: string;
  default_steps: number;
  default_guidance: number | null;
  quantize: number | null;
  scheduler: string;
  supports_guidance: boolean;
  supports_negative_prompt: boolean;
  supports_image_to_image: boolean;
  supports_edit: boolean;
};

export type Capabilities = {
  default_model: string;
  max_n: number;
  response_formats: string[];
  models: Record<string, ModelCapabilities>;
};

/** Snapshot streamed by `/v1/progress`. */
export type Progress = {
  state: "idle" | "loading" | "generating";
  model: string | null;
  kind: string | null;
  seed: number | null;
  step: number;
  total: number;
  elapsed_s: number | null;
  loaded_model: string | null;
  memory: Health["memory"];
};

/** Structured event read from the server's stdout, in `log_json` mode. */
export type LogEvent = {
  ts: string;
  level: string;
  logger: string;
  message: string;
  event?: string;
  fields?: Record<string, unknown>;
};

/** OpenAI-shaped error returned by the server. */
export type ApiError = {
  error: { message: string; type: string; param: string | null; code: string | null };
};
