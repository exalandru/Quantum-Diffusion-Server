/** Types partagés avec le Rust et avec l'API HTTP du serveur. */

// ── Côté Rust ──────────────────────────────────────────────────────────────

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

/** Une ligne de sortie du serveur. `structured` ⇒ stdout, donc du JSON. */
export type ServerLine = { structured: boolean; line: string };

// ── Côté serveur HTTP ──────────────────────────────────────────────────────

export type Health = {
  status: string;
  version: string;
  default_model: string;
  models: string[];
  loaded_model: string | null;
  /** Vide si `mlx.core` est introuvable : traiter les clés comme optionnelles. */
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

/** Instantané diffusé par `/v1/progress`. */
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

/** Événement structuré lu sur stdout du serveur, en mode `log_json`. */
export type LogEvent = {
  ts: string;
  level: string;
  logger: string;
  message: string;
  event?: string;
  fields?: Record<string, unknown>;
};

/** Erreur au format OpenAI renvoyée par le serveur. */
export type ApiError = {
  error: { message: string; type: string; param: string | null; code: string | null };
};
