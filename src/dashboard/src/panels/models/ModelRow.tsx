import { useState } from "react";

import { ActionNote } from "../../actions";
import { ModelSettings } from "./ModelSettings";
import { QuantizationDialog } from "./QuantizationDialog";
import { SizeBadge } from "./SizeBadge";
import { Switch } from "./Switch";
import { ACTION } from "./shared";
import type { RowProps } from "./shared";

/**
 * One model, read top to bottom: what it is, what state its weights are in,
 * where it comes from, and what can be done to it.
 *
 * Four bands, in that order, because they answer four different questions and
 * used to answer them on one baseline: the name sat in a row of pills, so
 * scanning ten models for the one you wanted meant reading past forty badges.
 * The name is now a heading with nothing beside it but its own switch; the state
 * is a band under it; the source is a band under that; the actions are last.
 *
 * The technical controls are behind two dialogs rather than in the row. A bit
 * depth selector and a Pre-quantize button in a catalogue row invite the reading
 * that they are part of installing it, which is how someone starts a two-hour
 * conversion meaning to download a model.
 *
 * Every branch below reads a backend field. Nothing re-derives whether a download
 * is offered (`can_download`), which bit depths exist (`prequantize_choices`),
 * whether conversion is possible (`supports_prequantize`) or which variant is
 * live (`active_variant`).
 */
export function ModelRow({
  model,
  caps,
  isDefault,
  tokenPresent,
  anyBusy,
  job,
  jobActive,
  cancelBusy,
  onCancelJob,
  busy,
  stateOf,
  dismiss,
  override,
  servedByServer,
  bits,
  onBits,
  onDownload,
  onConvert,
  onActivate,
  onForget,
  onPatch,
  located,
  onLocate,
  onResetLocation,
}: RowProps) {
  const [dialog, setDialog] = useState<"quantization" | "settings" | null>(null);
  const action = ACTION[model.availability];
  // A gated repository with no token would 401 several minutes in: better to say
  // so before starting. Gating is a HuggingFace concept, so an imported local
  // model — which has no repository — can never be blocked by it.
  const blocked = model.can_download && model.gated && !tokenPresent;
  const quant = model.quantization;
  const imported = model.provenance === "imported_local";
  const enabled = override.enabled !== false;

  // Two backend facts disagreeing, not a flag of our own: the configuration says
  // this model should be exposed — and which representation it should use — while
  // `/v1/capabilities` reports what the *running* server actually loaded its
  // registry with. That is exactly what a restart would fix, and it is the only
  // thing said about it.
  //
  // The variant half matters most right after a conversion: the supervisor
  // selects the new copy the moment it validates, and a running process cannot
  // have adopted a choice made after it started. Without this comparison the row
  // would show the new variant as active while generation kept using the old
  // representation.
  // A server that does not publish `active_variant` at all — an older
  // environment than this app — has told us nothing, and nothing is not a
  // disagreement. Present-and-null is a real answer ("the source") and is
  // compared; absent is left alone rather than turned into a permanent and false
  // "restart required".
  const liveVariant =
    caps !== undefined && "active_variant" in caps ? caps.active_variant : undefined;
  const restartRequired =
    servedByServer !== null &&
    (servedByServer !== enabled ||
      (liveVariant !== undefined && liveVariant !== model.active_variant));

  // Nothing to say about precision for a model that can neither be quantized on
  // load nor pre-quantized, and that has no saved copy to activate.
  const hasQuantization =
    quant.supports_quantization || quant.supports_prequantize || model.variants.length > 0;

  // The one fact from inside the dialog worth carrying in the row: which
  // representation generation will actually use.
  const variantSummary =
    model.active_variant !== null
      ? `using the ${model.active_variant}-bit copy`
      : model.variants.length > 0
        ? `${model.variants.length} saved ${model.variants.length === 1 ? "copy" : "copies"}`
        : null;

  return (
    <li className={`model${enabled ? "" : " off"}`}>
      {/* ── Identity ─────────────────────────────────────────────────────── */}
      <div className="model-head">
        <h3 className="model-name">{model.display_name}</h3>
        <span className="model-enable">
          <span className="label">{enabled ? "Enabled" : "Disabled"}</span>
          <Switch
            checked={enabled}
            label={`Enable ${model.display_name}`}
            disabled={anyBusy}
            onChange={(next) => onPatch({ enabled: next }, `enable:${model.key}`)}
          />
        </span>
      </div>

      {/* ── State ────────────────────────────────────────────────────────── */}
      <div className="model-badges">
        <span className={`pill ${action.tone}`} title={model.detail ?? undefined}>
          {action.badge}
        </span>
        {/* The measured size, separate from the availability it used to replace:
            "12.5 GB" answered "how big", never "is it here". What it reports is
            the *active* representation — a model set to its 4-bit copy loads
            5.9 GB, whatever its 20.5 GB source still occupies — and the rest of
            the accounting is a disclosure away rather than absent. */}
        <SizeBadge disk={model.disk} activeVariant={model.active_variant} />
        {model.availability !== "present" && model.size_gb > 0 && (
          <span className="pill" title="Size of the repository on Hugging Face, not on this disk.">
            {model.size_gb} GB to download
          </span>
        )}
        {isDefault && <span className="pill pill-accent">default</span>}
        {model.gated && <span className="pill pill-warn">gated</span>}
        {imported && <span className="pill">imported local</span>}
        {/* What this model *is* — weights that already carry their precision, so
            a runtime quantize setting would be ignored. */}
        {!quant.supports_quantization && quant.note && (
          <span className="pill" title={quant.note}>
            fixed precision
          </span>
        )}
        {restartRequired && (
          <span
            className="pill pill-warn"
            title="Saved. The running generation server still has its previous model set."
          >
            restart required
          </span>
        )}
      </div>

      {/* ── Source ───────────────────────────────────────────────────────── */}
      {/* For a built-in that is its repository; for an imported model, where it
          actually lives. Both are the source identity. */}
      <p className="model-meta">
        <code>{model.repo}</code>
        {/* Only where nothing else says it. A catalogue row now sits inside a
            fieldset legended with its release, which is the same fact in the
            words a reader uses — "Stable Diffusion 3.5" over three rows already
            labelled `sd35`. An imported model has no legend above it, and there
            the detected family is the one thing that says what the directory
            turned out to hold. */}
        {imported && model.family && (
          <>
            <span className="sep">·</span>
            <span>{model.family}</span>
          </>
        )}
        {imported ? (
          <>
            {/* What to put in `{"model": ...}`. The internal id is deliberately
                not shown as the thing to send. */}
            <span className="sep">·</span>
            <span>
              API name <code>{model.api_name}</code>
            </span>
            {model.base_profile_key && (
              <>
                <span className="sep">·</span>
                <span>defaults from {model.base_profile_key}</span>
              </>
            )}
          </>
        ) : model.license ? (
          <>
            <span className="sep">·</span>
            <span>{model.license}</span>
          </>
        ) : null}
        {caps && (
          <>
            <span className="sep">·</span>
            <span>
              {caps.default_steps} steps
              {caps.supports_guidance && caps.default_guidance !== null
                ? ` · guidance ${caps.default_guidance}`
                : ""}
              {caps.quantize ? ` · ${caps.quantize}-bit` : ""}
            </span>
          </>
        )}
      </p>

      {/* The server's own words. Paraphrasing loses the part that says what to do
          about it — an unplugged volume in particular. */}
      {model.detail && model.availability !== "present" && (
        <p className="library-detail">{model.detail}</p>
      )}

      {/* ── Actions ──────────────────────────────────────────────────────── */}
      <div className="model-actions">
        {/* A local artifact has nothing to fetch: its path either holds a
            completed artifact or it does not. */}
        {action.label && model.can_download && (
          <button
            className="small"
            onClick={onDownload}
            disabled={anyBusy || jobActive}
            title={
              blocked
                ? "This repository is gated: add a Hugging Face token in Configuration first, or the download will fail with a 401."
                : jobActive
                  ? "Another long operation is already running."
                  : (model.detail ?? undefined)
            }
          >
            {busy(`download:${model.key}`)
              ? "Starting…"
              : blocked
                ? `${action.label} ⚠`
                : action.label}
          </button>
        )}

        {!imported && action.label && model.can_download && (
          <button className="small" onClick={onLocate} disabled={anyBusy || jobActive}>
            {busy(`locate:${model.key}`) ? "Checking…" : "Locate…"}
          </button>
        )}

        {!imported && located && (
          <button
            className="small"
            onClick={onResetLocation}
            disabled={anyBusy}
            title="Forgets the folder and goes back to the catalogue repository. No files are deleted."
          >
            {busy(`locate:${model.key}`) ? "Resetting…" : "Reset location"}
          </button>
        )}

        {hasQuantization && (
          <button className="small" onClick={() => setDialog("quantization")}>
            Quantization…
          </button>
        )}

        <button className="small" onClick={() => setDialog("settings")}>
          Model settings…
        </button>

        {imported && (
          <button
            className="small"
            onClick={onForget}
            disabled={anyBusy}
            title="Removes the registration only. The model files are not deleted."
          >
            {busy(`forget:${model.key}`) ? "Removing…" : "Forget"}
          </button>
        )}

        {variantSummary && <span className="model-summary">{variantSummary}</span>}
      </div>

      {/* Only the outcomes of the actions this row still performs. A conversion's
          or a saved copy's belongs in the dialog that started it. */}
      {["download", "locate", "forget", "enable"].map((kind) => (
        <ActionNote
          key={kind}
          state={stateOf(`${kind}:${model.key}`)}
          onDismiss={() => dismiss(`${kind}:${model.key}`)}
        />
      ))}

      {dialog === "quantization" && (
        <QuantizationDialog
          model={model}
          override={override}
          anyBusy={anyBusy}
          job={job}
          jobActive={jobActive}
          cancelBusy={cancelBusy}
          onCancelJob={onCancelJob}
          busy={busy}
          stateOf={stateOf}
          dismiss={dismiss}
          bits={bits}
          onBits={onBits}
          onConvert={onConvert}
          onActivate={onActivate}
          onPatch={onPatch}
          onClose={() => setDialog(null)}
        />
      )}

      {dialog === "settings" && (
        <ModelSettings
          model={model}
          caps={caps}
          override={override}
          busy={busy(`settings:${model.key}`)}
          disabled={anyBusy}
          note={stateOf(`settings:${model.key}`)}
          onDismissNote={() => dismiss(`settings:${model.key}`)}
          onApply={(patch) =>
            onPatch(patch, `settings:${model.key}`, "Saved. It applies when the server restarts.")
          }
          onClose={() => setDialog(null)}
        />
      )}
    </li>
  );
}
