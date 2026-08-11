/** Types shared with the Rust side and with the server's HTTP API. */

// ── Rust side ──────────────────────────────────────────────────────────────

export type ServerStatus = {
  running: boolean;
  port: number | null;
  lastExit: string | null;
};

/**
 * What Setup must offer. The backend is the authority: React classifies nothing.
 *
 * `broken` is the state the old marker could not express. It lived inside the
 * directory the installer replaces, so quitting during the 1.1 GB `uv sync` left
 * new code with no marker — indistinguishable from a machine that had never been
 * installed, and offering "Install" when the truth was "repair this".
 */
export type BootstrapState =
  | "uninitialized"
  | "installing"
  | "updateRequired"
  | "broken"
  | "ready";

export type BootstrapStatus = {
  /** `state === "ready"`, kept as the one gate the whole app hangs on. */
  ready: boolean;
  state: BootstrapState;
  installedVersion: string | null;
  appVersion: string;
  envPath: string;
  /** Why the last install stopped, when it stopped. */
  failure: string | null;
};

export type Overview = {
  server: ServerStatus;
  bootstrap: BootstrapStatus;
  hfTokenPresent: boolean;
  hfHome: string;
  dataDir: string;
  configPath: string;
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
  /** Does the runtime quantize setting change anything for this model? */
  supports_quantization: boolean;
  /** Bit depths worth offering. Empty when unsupported — the app owns no list. */
  quantize_choices: number[];
  /** Can it be converted to a saved quantized artifact? (Slice 6 acts on this.) */
  supports_prequantize: boolean;
  prequantize_choices: number[];
  prequantize_strategy: "mflux_save" | "qds_memory_bounded" | null;
  /** Why a capability is off, for the UI to show instead of a dead control. */
  quantization_note: string | null;
  license: string;
  /** Repo requires approved access, so a download without a token would 401. */
  gated: boolean;
  /** `["json"]` means plain text is rejected outright. */
  prompt_formats: string[];
  preset: string | null;
  min_dimension: number;
  max_dimension: number | null;
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

/**
 * One catalogue entry with its weights' cache state, from `mflux-server-fetch
 * --status`. Comes through Rust rather than the HTTP API, so the list is complete
 * (disabled models included) and available with the server stopped.
 */
/**
 * Why a model's weights are or are not usable.
 *
 * Replaces a `cached: boolean` that collapsed five situations into one word and
 * got two of them dangerously wrong: an unreadable or unmounted cache reported
 * every model as "not downloaded" and offered to re-fetch tens of GB, and an
 * interrupted download reported as cached, hiding the only control that could
 * retry it.
 *
 * `present` means "in the cache with no on-disk evidence of an interrupted
 * download". It is deliberately not a claim that mflux can load it without
 * fetching anything more — see `availability.py` for why.
 */
export type Availability =
  | "present"
  | "partial"
  | "missing"
  | "volume_unmounted"
  | "unreadable";

/** The quantization contract, published per catalogue row. */
export type QuantizationCapability = {
  supports_quantization: boolean;
  quantize_choices: number[];
  supports_prequantize: boolean;
  prequantize_choices: number[];
  prequantize_strategy: "mflux_save" | "qds_memory_bounded" | null;
  note: string | null;
};

/** A saved, already-quantized copy of the model's current source. */
export type Variant = {
  bits: number;
  path: string;
  strategy: string | null;
  /** Recognised by the pre-marker rules rather than by a completion record. */
  legacy: boolean;
};

/** Python's verdict on a directory. Advisory: registration revalidates. */
export type ImportVerdict = {
  ok: boolean;
  path: string;
  availability: Availability | "invalid" | "incompatible";
  family: string | null;
  class_name: string | null;
  suggested_name: string | null;
  reason: string | null;
  profiles: string[];
  already_imported?: { id: string; display_name: string } | null;
  /** A free public identifier at the time of asking; `register` checks again. */
  suggested_api_name?: string | null;
};

/** Python's verdict on binding a directory to one built-in catalogue entry. */
export type LocateVerdict = {
  ok: boolean;
  path: string;
  model: string;
  availability: string;
  family: string | null;
  class_name: string | null;
  reason: string | null;
  /** The repository the path proves it came from, when it proves one. */
  detected_repo: string | null;
  /** False means "compatible, provenance unproven" — never "wrong". */
  repo_verified: boolean;
};

export type ModelStatus = {
  key: string;
  repo: string;
  license: string;
  gated: boolean;
  enabled: boolean;
  availability: Availability;
  /** Why, when the state alone does not say enough. */
  detail: string | null;
  /** A filesystem path rather than an HF repo: how availability is measured. */
  local: boolean;
  /** Where the entry came from. Authoritative for HF behaviour. */
  provenance: "built_in" | "imported_local";
  display_name: string;
  /** What an API request must send. A built-in's key; an imported model's alias. */
  api_name: string;
  base_profile_key: string | null;
  family: string | null;
  /** The single authority for offering Install/Resume. */
  can_download: boolean;
  size_gb: number;
  files: number;
  quantization: QuantizationCapability;
  /** Saved variants that exist and validate for the CURRENT source. */
  variants: Variant[];
  /** The bit depth generation is set to use, or null for the source itself. */
  active_variant: number | null;
};

/**
 * The long model operation Rust owns: a weight download or the FLUX.2-dev
 * conversion.
 *
 * Rust is the authority for this, not React. The panel used to hold the running
 * state in `useState`, and `App` unmounts it on a tab switch — so leaving the tab
 * lost the operation and re-armed its button while the child was still running.
 */
export type JobStatus = {
  state: "idle" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  kind: "fetch" | "prequantize" | null;
  /** Model key for a fetch; the component list for a conversion. */
  target: string | null;
  /** Last structured event name from the child's stdout, verbatim. */
  event: string | null;
  /** Its `fields`, verbatim — the child's schema, not a reshaped one. */
  fields: Record<string, unknown> | null;
  /** Current step while running; the terminal reason once finished. */
  message: string | null;
  startedAtMs: number | null;
  finishedAtMs: number | null;
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
