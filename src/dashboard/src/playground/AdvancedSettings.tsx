import type { ModelCapabilities } from "../types";

/** Long-side choices, in pixels. The short side follows the aspect ratio. */
export const RESOLUTIONS = [512, 1024, 1280, 1440, 1920, 2048, 3840] as const;

/** Stored landscape-first; the label flips with the orientation. */
export const RATIOS = {
  "1:1": 1,
  "5:4": 5 / 4,
  "4:3": 4 / 3,
  "3:2": 3 / 2,
  "16:9": 16 / 9,
  "21:9": 21 / 9,
} as const;

export type Advanced = {
  ratio: "auto" | keyof typeof RATIOS;
  orientation: "landscape" | "portrait";
  /** The long side; one of `RESOLUTIONS`. */
  long: number;
  /** null = the model's default step count. */
  steps: number | null;
  /** null = a random seed, picked by the server. */
  seed: number | null;
};

export const MAX_SEED = 4294967295;

export const DEFAULT_ADVANCED: Advanced = {
  ratio: "auto",
  orientation: "landscape",
  long: 1280,
  steps: null,
  seed: null,
};

/** "WxH" for the server, or null when the ratio is auto (model default size). */
export function sizeOf(a: Advanced): string | null {
  if (a.ratio === "auto") return null;
  const short = Math.round(a.long / RATIOS[a.ratio]);
  return a.orientation === "portrait" ? `${short}x${a.long}` : `${a.long}x${short}`;
}

/**
 * The ratio as the user reads it: portrait names the short side first, so 16:9
 * is 9:16 there. Only the label turns — the stored key stays landscape-first,
 * which is what `RATIOS` divides by.
 */
export function ratioLabel(ratio: keyof typeof RATIOS, orientation: Advanced["orientation"]) {
  if (orientation === "landscape" || ratio === "1:1") return ratio;
  const [long, short] = ratio.split(":");
  return `${short}:${long}`;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * The composer's advanced parameters, as a popover.
 *
 * Every change applies immediately — the values are only read when Generate is
 * pressed, so an Apply button would be a second source of truth. The server
 * remains the authority on what it accepts: out-of-range sizes are flagged
 * here, never blocked.
 */
export function AdvancedSettings({
  value,
  onChange,
  capabilities,
}: {
  value: Advanced;
  onChange: (next: Advanced) => void;
  capabilities: ModelCapabilities | null;
}) {
  const explicit = value.ratio !== "auto";
  const size = sizeOf(value);
  const short = value.ratio === "auto" ? null : Math.round(value.long / RATIOS[value.ratio]);
  const outOfRange =
    capabilities !== null &&
    short !== null &&
    (short < capabilities.min_dimension ||
      (capabilities.max_dimension !== null && value.long > capabilities.max_dimension));

  return (
    <div className="pg-popover" role="dialog" aria-label="Advanced settings">
      <div className="pg-field">
        <span className="pg-field-label">Aspect ratio</span>
        <div className="pg-seg" role="group" aria-label="Aspect ratio">
          <button
            type="button"
            aria-pressed={value.ratio === "auto"}
            onClick={() => onChange({ ...value, ratio: "auto" })}
          >
            Auto
          </button>
          {(Object.keys(RATIOS) as (keyof typeof RATIOS)[]).map((ratio) => (
            <button
              key={ratio}
              type="button"
              aria-pressed={value.ratio === ratio}
              onClick={() => onChange({ ...value, ratio })}
            >
              {ratioLabel(ratio, value.orientation)}
            </button>
          ))}
        </div>
        <p className="note">The image's proportions. Auto keeps the model's default size.</p>
      </div>

      <div className="pg-field">
        <span className="pg-field-label">Orientation</span>
        <div className="pg-seg" role="group" aria-label="Orientation">
          {(["landscape", "portrait"] as const).map((orientation) => (
            <button
              key={orientation}
              type="button"
              disabled={!explicit || value.ratio === "1:1"}
              aria-pressed={value.orientation === orientation}
              onClick={() => onChange({ ...value, orientation })}
            >
              {orientation === "landscape" ? "Landscape" : "Portrait"}
            </button>
          ))}
        </div>
        <p className="note">Which side is the long one. No effect on a square.</p>
      </div>

      <div className="pg-field">
        <span className="pg-field-label">Resolution</span>
        <div className="pg-seg" role="group" aria-label="Resolution">
          {RESOLUTIONS.map((long) => (
            <button
              key={long}
              type="button"
              disabled={!explicit}
              aria-pressed={value.long === long}
              onClick={() => onChange({ ...value, long })}
            >
              {long}
            </button>
          ))}
        </div>
        <p className="note">
          The long side, in pixels - the short side follows the aspect ratio. Larger is sharper
          and slower.
        </p>
        {size && <p className="pg-size-readout">→ {size.replace("x", " × ")}</p>}
        {outOfRange && capabilities && (
          <p className="caution">
            Outside this model's range [{capabilities.min_dimension},{" "}
            {capabilities.max_dimension ?? "∞"}] - the server will refuse it.
          </p>
        )}
      </div>

      <div className="pg-field">
        <span className="pg-field-label">
          <label htmlFor="pg-steps">Steps</label>
        </span>
        <input
          id="pg-steps"
          type="number"
          min={1}
          max={100}
          value={value.steps ?? ""}
          placeholder={`auto (${capabilities?.default_steps ?? "model default"})`}
          onChange={(event) => {
            const raw = event.target.value;
            onChange({
              ...value,
              steps: raw === "" ? null : clamp(Math.round(Number(raw)), 1, 100),
            });
          }}
        />
        <p className="note">
          Denoising passes - more recovers detail and costs time linearly. Empty uses the model's
          default.
        </p>
      </div>

      <div className="pg-field">
        <span className="pg-field-label">
          <label htmlFor="pg-seed">Seed</label>
        </span>
        <input
          id="pg-seed"
          type="number"
          min={0}
          max={MAX_SEED}
          value={value.seed ?? ""}
          placeholder="auto"
          onChange={(event) => {
            const raw = event.target.value;
            onChange({
              ...value,
              seed: raw === "" ? null : clamp(Math.round(Number(raw)), 0, MAX_SEED),
            });
          }}
        />
        <p className="note">
          The same seed and prompt reproduce the same image. Empty picks one at random.
        </p>
      </div>
    </div>
  );
}
