import type { ActionNote } from "../../actions";
import type {
  Availability,
  DiskEntry,
  JobStatus,
  ModelCapabilities,
  ModelStatus,
} from "../../types";

/**
 * How each availability is *presented*, and what it must not offer.
 *
 * A label-and-tone map, not a rule: whether the Install button may appear at all
 * is `can_download`, which the backend decides. The one thing worth restating is
 * why `volume_unmounted` and `unreadable` carry no label — the weights are not
 * gone, the disk holding them is unplugged or unreadable, and offering "Install"
 * there would invite a re-download of tens of gigabytes the user already owns.
 */
export const ACTION: Record<Availability, { label: string | null; badge: string; tone: string }> = {
  present: { label: null, badge: "installed", tone: "pill-ok" },
  partial: { label: "Resume", badge: "incomplete", tone: "pill-warn" },
  missing: { label: "Install", badge: "not installed", tone: "pill" },
  volume_unmounted: { label: null, badge: "volume unavailable", tone: "pill-warn" },
  unreadable: { label: null, badge: "unreadable", tone: "pill-bad" },
};

/** Bytes as a figure someone reads off a disk, or nothing when it is unknown. */
export function formatBytes(bytes: number | null | undefined): string | null {
  if (bytes === null || bytes === undefined) return null;
  if (bytes <= 0) return null;
  return `${(bytes / 1e9).toFixed(bytes < 1e10 ? 2 : 1)} GB`;
}

/**
 * Which components of one bit depth are already converted.
 *
 * Three states collapse into two here, and deliberately: a *complete* variant
 * has every component by definition, a partial conversion carries the backend's
 * per-component judgement, and anything else has none. Nothing is inferred from
 * the existence of a directory — `components` is what validation found on disk.
 */
export function componentStates(model: ModelStatus, bits: number): Record<string, string> {
  if (model.variants.some((variant) => variant.bits === bits)) {
    return Object.fromEntries(
      model.quantization.prequantize_components.map((spec) => [spec.key, "complete"]),
    );
  }
  return model.partials.find((partial) => partial.bits === bits)?.components ?? {};
}

/** What one entry in the disk breakdown is called. Bits come from the backend. */
export function describeEntry(entry: DiskEntry): string {
  if (entry.kind === "source") return "Original";
  const bits = entry.bits === null ? "" : `${entry.bits}-bit `;
  if (entry.kind === "partial") return `${bits}conversion in progress`;
  return entry.is_source ? `${bits}artifact (this model's source)` : `${bits}variant`;
}

/**
 * The two halves of the built-in catalogue, as tabs.
 *
 * A table rather than two branches: the tablist, the counts and the shown list
 * all read the same row, so a tab cannot end up labelled one thing and listing
 * another. `models` picks from the already-filtered groups — the split itself is
 * the backend's `gated`, never recomputed here.
 */
export type CatalogueTab = {
  id: "open" | "gated";
  label: string;
  heading: string;
  note: string;
  models: (groups: { gated: ModelStatus[]; ungated: ModelStatus[] }) => ModelStatus[];
};

/** The half that opens: installable right now, no account and nothing to request. */
export const OPEN_TAB: CatalogueTab = {
  id: "open",
  label: "Open",
  heading: "Open models",
  note: "Downloadable with no account and no token.",
  models: ({ ungated }) => ungated,
};

export const GATED_TAB: CatalogueTab = {
  id: "gated",
  label: "Gated",
  heading: "Gated models",
  note: "Access is granted per repository on its Hugging Face model card, and a token proves it.",
  models: ({ gated }) => gated,
};

export const CATALOGUE_TABS: readonly CatalogueTab[] = [OPEN_TAB, GATED_TAB];

/** The backend's last published choice, which is its widest. Never a local list. */
export function lastChoice(model: ModelStatus): number {
  const choices = model.quantization.prequantize_choices;
  return choices[choices.length - 1] ?? 8;
}

export type RowProps = {
  model: ModelStatus;
  caps: ModelCapabilities | undefined;
  isDefault: boolean;
  tokenPresent: boolean;
  anyBusy: boolean;
  /** The server-owned operation, so a conversion can be watched where it started. */
  job: JobStatus | null;
  jobActive: boolean;
  cancelBusy: boolean;
  onCancelJob: () => void;
  busy: (key: string) => boolean;
  stateOf: (key: string) => Parameters<typeof ActionNote>[0]["state"];
  dismiss: (key: string) => void;
  /** This model's entry in the configuration, or an empty one. */
  override: Record<string, any>;
  /** The running server exposes this model. `null` when there is no server to ask. */
  servedByServer: boolean | null;
  bits: number;
  onBits: (value: number) => void;
  onDownload: () => void;
  onConvert: (bits: number, components: string[]) => void;
  onActivate: (bits: number | null) => void;
  onForget: () => void;
  onPatch: (patch: Record<string, unknown>, key: string, success?: string) => void;
  /** This built-in reads from a local `model_path` override. */
  located: boolean;
  onLocate: () => void;
  onResetLocation: () => void;
};

/** The same conservative slug the backend derives, for the live default. */
export function slugFor(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "")
    .slice(0, 64);
}
