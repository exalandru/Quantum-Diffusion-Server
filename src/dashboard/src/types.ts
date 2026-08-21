/** Types shared with the server's HTTP API. */

// ── Control plane ──────────────────────────────────────────────────────────

/**
 * What the server can say about itself that is not generation.
 *
 * Smaller than the desktop app's `Overview`, and the missing halves say where
 * the responsibilities went: there is no `server.running` because this page
 * only exists when the server runs, and no `bootstrap` because installing the
 * environment belongs to the menubar app, which renders none of this.
 */
export type Overview = {
  version: string;
  configPath: string;
  dataDir: string;
  effectiveHfHome: string;
  effectiveCacheDir: string;
  hfTokenPresent: boolean;
  /** The configuration failed to load; only the repair surface is served. */
  recoveryMode: boolean;
  recoveryError: string | null;
  /** The file on disk now says something this process did not read at start. */
  restartRequired: boolean;
  /** Whether the control plane is protected. The network toggle needs it. */
  adminPasswordSet: boolean;
  /** This machine's own addresses, to hand out when listening on the network. */
  lanAddresses: string[];
  server: { host: string; port: number };
};

/**
 * The configuration as the server hands it back.
 *
 * Either the document, or — when the file will not parse — its text and the
 * reason. The second case is not hypothetical: it is exactly what puts the
 * server in recovery mode, and therefore what this screen exists to repair.
 */
export type ConfigDocument =
  | (Record<string, unknown> & { unparsed?: undefined })
  | { unparsed: true; reason: string; text: string };

/** One page of the server's log tail. See `logsAfter`. */
export type LogPage = {
  entries: LogEntry[];
  lastSeq: number;
  /** Lines that fell out of the ring buffer between polls. */
  dropped: number;
};

export type LogEntry = {
  seq: number;
  ts: string;
  level: string;
  logger: string;
  message: string;
  event?: string;
  fields?: Record<string, unknown>;
};

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
  /**
   * The saved variant *this running process* loaded, or null for the source.
   *
   * Compared against the catalogue's `active_variant`, which is what the
   * configuration currently selects. They differ exactly when a restart is
   * needed for the live process to use what has been chosen.
   */
  active_variant: number | null;
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

/** One entry of `/v1/models`, with its capabilities under `mflux`. */
export type ModelEntry = {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  mflux: ModelCapabilities;
};

export type ModelList = { object: string; data: { id: string; display_name: string }[] };

/**
 * Whether this server offers prompt rewriting, and on what terms.
 *
 * Deliberately carries no model name: the person writing a prompt needs to know
 * it will be improved and what a first use costs, not which LLM does it.
 */
export type RewriteCapabilities = {
  available: boolean;
  /** Why it is off, when it is. Null when available. */
  reason: string | null;
  /**
   * Whether a first Enhance avoids a download. Asked of the files, not the
   * repo, exactly as `Upscaler.downloaded` is.
   */
  downloaded: boolean;
  /** Download size in megabytes, or null when unavailable. */
  sizeMb: number | null;
  /**
   * At or above this many words, the server generates the prompt as typed.
   *
   * Published rather than duplicated as a constant here, so the composer can
   * say so *before* submitting and cannot drift from what the route enforces.
   */
  word_ceiling: number;
};

export type Capabilities = {
  default_model: string;
  max_n: number;
  response_formats: string[];
  models: Record<string, ModelCapabilities>;
  rewrite: RewriteCapabilities;
};

/** Snapshot streamed by `/v1/progress`. */
export type Progress = {
  state: "idle" | "loading" | "generating" | "upscaling" | "rewriting";
  model: string | null;
  kind: string | null;
  seed: number | null;
  step: number;
  total: number;
  /**
   * Monotonic id of the latest preview frame at `/playground/api/preview`;
   * 0 when there is none.
   */
  preview_seq: number;
  elapsed_s: number | null;
  loaded_model: string | null;
  /**
   * The resident upscaler's key, or null. Optional so the several `IDLE`
   * literals in the app and its tests keep type-checking unchanged.
   */
  upscaler?: string | null;
  memory: Health["memory"];
};

/** A playground conversation, as `/playground/api/sessions` lists it. */
export type PlaygroundSession = {
  id: string;
  /** The name the user gave it, else the first prompt, truncated. Null until either. */
  title: string | null;
  createdAt: number;
  updatedAt: number;
  /** Something in this session is queued or running: the sidebar's live dot. */
  generating: boolean;
  /** Has a password: its content is served only with an unlock token. */
  locked: boolean;
};

/**
 * What `/playground/api/sessions` answers.
 *
 * `paused` rides along with the list rather than on `/v1/progress`: holding the
 * queue is a playground control, and that stream is the engine's state, shared
 * with `/v1` clients who have no queue to hold.
 */
export type PlaygroundSessionList = {
  sessions: PlaygroundSession[];
  /** The playground's FIFO worker is holding: nothing new starts. Global. */
  paused: boolean;
};

/** One entry of the upscaler catalogue, as `/playground/api/upscalers` serves it. */
export type Upscaler = {
  id: string;
  name: string;
  scales: number[];
  /** Whether using it now avoids a download. Asked of the file, not the repo. */
  downloaded: boolean;
  sizeMb: number;
  license: string;
};

export type PlaygroundStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

/**
 * One durable generation record.
 *
 * Server-owned in full: closing the tab loses nothing, and reopening the page
 * rebuilds the feed from these alone.
 */
export type PlaygroundGeneration = {
  id: string;
  sessionId: string;
  /**
   * The lineage this generation belongs to — its own id unless it was started
   * from another generation's image. The feed renders one group as one entry.
   */
  groupId: string;
  prompt: string;
  /**
   * What the image was told to avoid, or null when none was sent.
   *
   * Null for every model whose pipeline has no unconditional branch to apply one
   * to: the composer greys the field out for those, and the server refuses it.
   */
  negativePrompt: string | null;
  /**
   * What the rewriter made of `prompt`, or null if nothing rewrote it.
   *
   * `prompt` above is never overwritten, so the feed can title an entry with
   * what the user actually typed and show this behind a disclosure. A variation
   * carries this forward rather than asking for a fresh rewrite: a rewrite is
   * sampled, so re-running it would produce different words and the result
   * would not be a variation of anything.
   */
  rewrittenPrompt: string | null;
  /**
   * Why a requested rewrite did not happen, or null.
   *
   * Not redundant with `rewrittenPrompt === null`: without it, "the rewriter
   * failed and we generated from your prompt" and "no rewrite was asked for"
   * look identical, and the first is something the user is owed a line about.
   */
  rewriteError: string | null;
  /** Public model name, as `/v1/models` lists it. */
  model: string;
  kind: "txt2img" | "edit" | "upscale";
  n: number;
  size: string;
  steps: number;
  seeds: number[];
  /** Root-relative URL of the reference image, when one was attached. */
  contextImage: string | null;
  status: PlaygroundStatus;
  error: string | null;
  images: { url: string; seed: number }[];
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
};

/**
 * One catalogue entry with its weights' cache state, from `/admin/models`.
 *
 * The same view `qds fetch --status` prints, computed in the server process.
 * The list is complete — disabled models included — because this is model
 * *management*, and a model you cannot see is a model you cannot enable.
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

/**
 * One independently convertible part of a model, as the backend describes it.
 *
 * The app owns no family→component table. It used to: three FLUX.2-dev names in
 * a `const` in `Models.tsx`, shown for whatever model reached that branch.
 */
export type ComponentSpec = {
  /** mflux's own name for it, which is also what a conversion request sends. */
  key: string;
  label: string;
  /** Part of the set a usable artifact must carry. */
  required: boolean;
  independently_convertible: boolean;
  /** False when the family declines to quantize it: saved, but not smaller. */
  quantized: boolean;
  note: string | null;
};

/** The quantization contract, published per catalogue row. */
export type QuantizationCapability = {
  supports_quantization: boolean;
  quantize_choices: number[];
  supports_prequantize: boolean;
  prequantize_choices: number[];
  prequantize_strategy: "mflux_save" | "qds_memory_bounded" | null;
  /** Empty for a family whose components have not been established. */
  prequantize_components: ComponentSpec[];
  /**
   * What this model loads at when the config asks for nothing — the catalogue's
   * own choice, `null` meaning bf16. Shown in the runtime selector so "default"
   * names a depth instead of hiding one.
   */
  catalogue_quantize: number | null;
  note: string | null;
};

/** A saved, already-quantized copy of the model's current source. */
export type Variant = {
  bits: number;
  path: string;
  strategy: string | null;
  /** Recognised by the pre-marker rules rather than by a completion record. */
  legacy: boolean;
  /** Bytes it actually occupies. `null` when nothing measured it. */
  size_bytes: number | null;
};

/**
 * Components converted towards a variant that does not exist yet.
 *
 * Deliberately a separate list from `variants`: nothing here may be activated,
 * and merging the two is exactly how a half-converted model would come to be
 * offered as usable.
 */
export type PartialConversion = {
  bits: number;
  path: string;
  strategy: string | null;
  /** Component key → `"complete"` / `"missing"`, judged from disk. */
  components: Record<string, string>;
  size_bytes: number;
};

/** One thing on disk attributable to a model. Deduplicated by path upstream. */
export type DiskEntry = {
  kind: "source" | "variant" | "partial";
  bits: number | null;
  bytes: number;
  path: string;
  /** This directory is also the model's source — FLUX.2-dev's 8-bit artifact. */
  is_source: boolean;
};

/**
 * Three different questions about storage, kept apart.
 *
 * `null` means unknown — a source that is not on this machine has no disk usage,
 * and its catalogue size is not an answer to how much room it is taking.
 */
export type DiskReport = {
  source_bytes: number | null;
  /** What generation will actually load: the active variant, or the source. */
  active_bytes: number | null;
  total_bytes: number;
  breakdown: DiskEntry[];
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
  /** Conversions in progress towards a variant. Never activatable. */
  partials: PartialConversion[];
  /** The bit depth generation is set to use, or null for the source itself. */
  active_variant: number | null;
  disk: DiskReport;
};

/**
 * A runtime invariant this configuration breaks, named by the backend.
 *
 * Not an error: the catalogue is perfectly readable, and these are the reasons
 * the generation server would refuse to start. They travel with the rows so the
 * view that can repair them is also the view that reports them.
 */
export type ConfigWarning = { code: string; field: string; message: string };

/** What one catalogue read answers: the rows, and what is wrong with the config. */
export type CatalogueStatus = { models: ModelStatus[]; warnings: ConfigWarning[] };

/**
 * The long model operation the server owns: a weight download or a
 * conversion.
 *
 * The server is the authority for this, not React. The panel used to hold the running
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
