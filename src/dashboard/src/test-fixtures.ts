/**
 * Backend-shaped fixtures.
 *
 * Every field here exists because the backend publishes it. The point of the
 * tests that use them is that the interface renders what the backend says and
 * decides nothing itself — so a fixture that invented a field, or that omitted
 * one the components read, would defeat the exercise.
 */
import type {
  CatalogueStatus,
  ComponentSpec,
  ConfigWarning,
  DiskReport,
  JobStatus,
  ModelStatus,
  Overview,
  PlaygroundSession,
} from "./types";

export function playgroundSession(patch: Partial<PlaygroundSession> = {}): PlaygroundSession {
  return {
    id: "s1",
    title: "a fox",
    createdAt: 1_700_000_000,
    updatedAt: 1_700_000_000,
    generating: false,
    locked: false,
    // No cover by default: a project with no images is the fixture's starting
    // point, and it is also what a locked project publishes.
    cover: null,
    ...patch,
  };
}

export function overview(patch: Partial<Overview> = {}): Overview {
  return {
    version: "2.0.0",
    server: { host: "127.0.0.1", port: 8765 },
    hfTokenPresent: true,
    effectiveHfHome: "/hf",
    effectiveCacheDir: "/data/cache",
    dataDir: "/data",
    configPath: "/data/server-config.json",
    recoveryMode: false,
    recoveryError: null,
    restartRequired: false,
    adminPasswordSet: false,
    playgroundPasswordSet: false,
    lanAddresses: ["192.168.1.19"],
    ...patch,
  };
}

export function model(patch: Partial<ModelStatus> = {}): ModelStatus {
  return {
    key: "z-image-turbo",
    repo: "mlx-community/Z-Image-Turbo-bf16",
    license: "Apache-2.0",
    gated: false,
    enabled: true,
    availability: "present",
    detail: null,
    local: false,
    provenance: "built_in",
    display_name: "z-image-turbo",
    api_name: "z-image-turbo",
    base_profile_key: null,
    family: "z-image",
    // Unlabelled by default, so a test that is about a row says nothing about
    // releases: the panel renders these as one plain list. The tests that *are*
    // about grouping set it themselves.
    group_label: null,
    can_download: true,
    size_gb: 12.5,
    files: 9,
    quantization: {
      supports_quantization: true,
      // Deliberately not the real catalogue's list. A fixture that mirrors the
      // backend invites being read as the backend, and `test_react_keeps_no_
      // quantization_table_of_its_own` rightly cannot tell a mirror from a table.
      quantize_choices: [4, 8],
      // Likewise a fixture value, not the catalogue's.
      catalogue_quantize: 8,
      supports_prequantize: false,
      prequantize_choices: [],
      prequantize_strategy: null,
      prequantize_components: [],
      note: null,
    },
    variants: [],
    partials: [],
    active_variant: null,
    disk: { source_bytes: null, active_bytes: null, total_bytes: 0, breakdown: [] },
    ...patch,
  };
}

/**
 * A model's components, as the backend publishes them.
 *
 * Deliberately not the real catalogue's three names in the real order: a fixture
 * that mirrors the backend can be mistaken for the backend, and the point of
 * every test using this is that the interface renders whatever list it is given.
 */
export function components(...keys: string[]): ComponentSpec[] {
  return keys.map((key) => ({
    key,
    label: key === "vae" ? "VAE" : key.replace("_", " ").replace(/^./, (c) => c.toUpperCase()),
    required: true,
    independently_convertible: true,
    quantized: true,
    note: null,
  }));
}

/** A disk report, so a row can say what it occupies without inventing a number. */
export function disk(patch: Partial<DiskReport> = {}): DiskReport {
  return {
    source_bytes: 20_000_000_000,
    active_bytes: 20_000_000_000,
    total_bytes: 20_000_000_000,
    breakdown: [
      {
        kind: "source",
        bits: null,
        bytes: 20_000_000_000,
        path: "/hf/models--x",
        is_source: true,
      },
    ],
    ...patch,
  };
}

export function job(patch: Partial<JobStatus> = {}): JobStatus {
  return {
    state: "idle",
    kind: null,
    target: null,
    event: null,
    fields: null,
    message: null,
    startedAtMs: null,
    finishedAtMs: null,
    ...patch,
  };
}

/**
 * One catalogue read: the rows, and what is wrong with the configuration.
 *
 * Two outputs because the backend has two answers. A configuration the
 * *generation server* would refuse — a default model that has been switched off
 * — used to make this call fail outright, which took the Models view down with
 * it and removed the only screen that could repair it.
 */
export function catalogue(
  models: ModelStatus[],
  warnings: ConfigWarning[] = [],
): CatalogueStatus {
  return { models, warnings };
}
