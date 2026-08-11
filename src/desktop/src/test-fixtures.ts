/**
 * Backend-shaped fixtures.
 *
 * Every field here exists because the backend publishes it. The point of the
 * tests that use them is that the interface renders what the backend says and
 * decides nothing itself — so a fixture that invented a field, or that omitted
 * one the components read, would defeat the exercise.
 */
import type { JobStatus, ModelStatus, Overview } from "./types";

export function overview(patch: Partial<Overview> = {}): Overview {
  return {
    server: { running: false, port: 8765, lastExit: null },
    bootstrap: {
      ready: true,
      state: "ready",
      installedVersion: "0.2.0",
      appVersion: "0.2.0",
      envPath: "/data/env",
      failure: null,
    },
    hfTokenPresent: true,
    hfHome: "/hf",
    dataDir: "/data",
    configPath: "/data/server-config.json",
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
    can_download: true,
    size_gb: 12.5,
    files: 9,
    quantization: {
      supports_quantization: true,
      // Deliberately not the real catalogue's list. A fixture that mirrors the
      // backend invites being read as the backend, and `test_react_keeps_no_
      // quantization_table_of_its_own` rightly cannot tell a mirror from a table.
      quantize_choices: [4, 8],
      supports_prequantize: false,
      prequantize_choices: [],
      prequantize_strategy: null,
      note: null,
    },
    variants: [],
    active_variant: null,
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
