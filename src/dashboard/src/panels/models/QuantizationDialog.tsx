import { useEffect, useRef, useState } from "react";

import { ActionNote } from "../../actions";
import { describeJob, describeOutcome } from "../../job";
import { Modal } from "../../modal";
import { componentStates } from "./shared";
import type { RowProps } from "./shared";

/**
 * Quantization, which is two mechanisms that share a word.
 *
 * They are rendered as two sections and never merged, because they differ in
 * every way that matters: one is applied while the model loads and leaves the
 * disk untouched, the other writes a second copy of the weights and is then a
 * thing you activate. A single "quantization" control would have to lie about
 * one of them — and the row's old inline selector, which drove only the second,
 * was read as driving both.
 *
 * Both choice lists are the backend's (`quantize_choices`, `prequantize_choices`)
 * and neither is defaulted here.
 */
export function QuantizationDialog({
  model,
  override,
  anyBusy,
  job,
  jobActive,
  cancelBusy,
  onCancelJob,
  busy,
  stateOf,
  dismiss,
  bits,
  onBits,
  onConvert,
  onActivate,
  onPatch,
  onClose,
}: Pick<
  RowProps,
  | "model"
  | "override"
  | "anyBusy"
  | "job"
  | "jobActive"
  | "cancelBusy"
  | "onCancelJob"
  | "busy"
  | "stateOf"
  | "dismiss"
  | "bits"
  | "onBits"
  | "onConvert"
  | "onActivate"
  | "onPatch"
> & { onClose: () => void }) {
  const quant = model.quantization;
  const configured = override.quantize;
  const runtimeValue = configured === null || configured === undefined ? "" : String(configured);
  // A stored value the backend no longer publishes is kept visible and marked,
  // rather than silently coerced.
  const stale =
    typeof configured === "number" &&
    configured !== 0 &&
    !quant.quantize_choices.includes(configured)
      ? configured
      : null;

  // What this model is made of, and what is already converted at the depth
  // currently selected. Both are the backend's: an empty list is a family whose
  // components nobody has established, and the correct response to that is to
  // offer no component controls rather than to guess three names.
  const componentSpecs = quant.prequantize_components;
  const states = componentStates(model, bits);
  const converted = componentSpecs.filter((spec) => states[spec.key] === "complete");
  const missing = componentSpecs.filter((spec) => states[spec.key] !== "complete");

  // Seeded with what is left to do, which is what "continue" means, and reset
  // whenever the depth changes because the answer is a different artifact's.
  const [selected, setSelected] = useState<string[]>(() => missing.map((spec) => spec.key));
  const seededFor = useRef(bits);
  useEffect(() => {
    if (seededFor.current !== bits) {
      seededFor.current = bits;
      setSelected(missing.map((spec) => spec.key));
    }
  }, [bits, missing]);

  // A component this dialog was offering to convert, which a finished run has
  // since converted, stops being a thing to select. Driven by the refreshed
  // status rather than by the job: what makes a component done is that the
  // artifact says so.
  const stateKey = componentSpecs
    .map((spec) => `${spec.key}:${states[spec.key] ?? "missing"}`)
    .join(",");
  useEffect(() => {
    setSelected((previous) => previous.filter((key) => !stateKey.includes(`${key}:complete`)));
  }, [stateKey]);

  // The child's own last word about this model, once the operation has settled.
  // Shown here as well as in the panel behind, which is under a backdrop while
  // this is open — a conversion that just finished must not be invisible to the
  // person who started it from this dialog.
  const finishedHere =
    !jobActive &&
    job?.state === "completed" &&
    (job.fields as { model?: string } | null)?.model === model.key
      ? describeOutcome(
          job,
          (component) => componentSpecs.find((spec) => spec.key === component)?.label ?? component,
        )
      : null;

  // OBSERVED: `start_prequantize` names the job `"{model} @ {bits}-bit"`. This is
  // a convenience mirror of the panel's own job section, which remains the place
  // the operation is reported whatever its label says: if that format ever
  // changed, this block would stop appearing rather than claim the wrong model.
  const converting =
    jobActive && job?.kind === "prequantize" && (job.target ?? "").startsWith(`${model.key} @`);

  return (
    <Modal
      title={`Quantization - ${model.display_name}`}
      subtitle={model.repo}
      onClose={onClose}
    >
      <fieldset className="settings-group">
        <legend>Runtime quantization</legend>
        <p className="setting-help" style={{ marginTop: 0 }}>
          Applied while the model loads, every time it loads. Nothing is written to disk and no
          second copy is made, so the cost is paid again on each load and clearing the setting
          returns to the weights exactly as they were downloaded.
        </p>

        {!quant.supports_quantization ? (
          <p className="setting-help">{quant.note ?? "Fixed for this model."}</p>
        ) : (
          <div className="setting" style={{ marginTop: 12 }}>
            <label className="setting-label" htmlFor={`runtime-quantize-${model.key}`}>
              Applied on load
            </label>
            <select
              id={`runtime-quantize-${model.key}`}
              aria-label={`Runtime quantization for ${model.key}`}
              value={runtimeValue}
              disabled={anyBusy}
              onChange={(event) =>
                onPatch(
                  { quantize: event.target.value === "" ? null : Number(event.target.value) },
                  `quantize:${model.key}`,
                  "Saved. It applies when the server restarts.",
                )
              }
            >
              <option value="">
                {quant.catalogue_quantize === null || quant.catalogue_quantize === undefined
                  ? "default (bf16)"
                  : `default (${quant.catalogue_quantize} bits)`}
              </option>
              <option value="0">none (bf16)</option>
              {quant.quantize_choices.map((choice) => (
                <option key={choice} value={String(choice)}>
                  {choice} bits
                </option>
              ))}
              {stale !== null && <option value={String(stale)}>{stale} bits (invalid)</option>}
            </select>
            {stale !== null && <p className="setting-error">not supported by this model</p>}
            <p className="setting-help">Applies when the generation server restarts.</p>
          </div>
        )}
        <ActionNote
          state={stateOf(`quantize:${model.key}`)}
          onDismiss={() => dismiss(`quantize:${model.key}`)}
        />
      </fieldset>

      <fieldset className="settings-group">
        <legend>Pre-quantized copy</legend>
        <p className="setting-help" style={{ marginTop: 0 }}>
          Writes a quantized copy of these weights to disk, once. QDS can then activate that copy and
          load it instead of the source, which is left where it is - both remain on disk, and
          switching between them is a setting rather than another conversion.
        </p>

        {!quant.supports_prequantize ? (
          <p className="setting-help">
            {quant.note ?? "This model cannot be converted to a saved quantized copy."}
          </p>
        ) : (
          <>
            <div className="row" style={{ marginTop: 12 }}>
              <select
                id={`bits-${model.key}`}
                aria-label={`Pre-quantized bit depth for ${model.key}`}
                value={String(bits)}
                onChange={(event) => onBits(Number(event.target.value))}
              >
                {quant.prequantize_choices.map((choice) => (
                  <option key={choice} value={String(choice)}>
                    {choice} bits
                  </option>
                ))}
              </select>
              <button
                className="small primary"
                onClick={() => onConvert(bits, selected)}
                disabled={anyBusy || jobActive || selected.length === 0}
                title={jobActive ? "Another long operation is already running." : undefined}
              >
                {busy(`convert:${model.key}`)
                  ? "Starting…"
                  : converted.length > 0
                    ? "Continue pre-quantization"
                    : "Pre-quantize selected"}
              </button>
            </div>

            {/* The model's own components, named by the backend. This used to be
                three FLUX.2-dev names in a constant in this file, shown for
                whatever model reached the branch. */}
            {componentSpecs.length > 0 && (
              <div className="setting" style={{ marginTop: 14 }}>
                <span className="setting-label" id={`components-${model.key}`}>
                  Components
                </span>
                <ul className="components" aria-labelledby={`components-${model.key}`}>
                  {componentSpecs.map((component) => {
                    const done = states[component.key] === "complete";
                    return (
                      <li className="component" key={component.key}>
                        {/* The whole identity is one label and one hit target:
                            box, name and note read left to right, and the status
                            is the only thing on the right. The name used to be
                            centred between them, which put "Text encoder" on two
                            lines in the middle of a row with space at both
                            ends. */}
                        <label className="check component-id">
                          <input
                            type="checkbox"
                            checked={selected.includes(component.key)}
                            disabled={anyBusy || jobActive}
                            onChange={(event) =>
                              setSelected((previous) =>
                                event.target.checked
                                  ? componentSpecs
                                      .filter(
                                        (item) =>
                                          previous.includes(item.key) || item.key === component.key,
                                      )
                                      .map((item) => item.key)
                                  : previous.filter((key) => key !== component.key),
                              )
                            }
                          />
                          <span className="component-name">{component.label}</span>
                          {!component.quantized && component.note && (
                            <span className="component-note" title={component.note}>
                              saved at source precision
                            </span>
                          )}
                        </label>
                        <span className={done ? "pill pill-ok" : "pill"}>
                          {done ? "Converted" : "Not converted"}
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <p className="setting-help">
                  Each component is loaded, quantized, written and released on its own, so the
                  memory needed is the largest single component rather than the whole model. They
                  can be done in separate runs - what is already converted is kept.
                </p>
                {missing.length > 0 && converted.length > 0 && (
                  <p className="caution">
                    {converted.length} of {componentSpecs.length} components converted. This{" "}
                    {bits}-bit copy cannot be used until {missing.map((s) => s.label).join(" and ")}{" "}
                    {missing.length === 1 ? "is" : "are"} converted too.
                  </p>
                )}
              </div>
            )}

            {converting && job && (
              <div className="job-inline">
                <div className="row">
                  <span className={job.state === "cancelling" ? "pill pill-warn" : "pill pill-live"}>
                    {job.state === "cancelling" ? "Stopping" : "Converting"}
                  </span>
                  <strong>{job.target}</strong>
                  <button
                    className="small danger"
                    onClick={onCancelJob}
                    disabled={job.state === "cancelling" || cancelBusy}
                  >
                    {job.state === "cancelling" ? "Stopping…" : "Cancel"}
                  </button>
                </div>
                <p className="note">{describeJob(job)}</p>
                {/* The child reports phases and blocks, never a byte total. */}
                <div className="bar bar-indeterminate" />
                {/* The panel behind this one carries the same note, and is
                    behind a backdrop while this is open: a refused cancel has to
                    appear where it was asked for. */}
                <ActionNote state={stateOf("cancel")} onDismiss={() => dismiss("cancel")} />
              </div>
            )}
          </>
        )}

        {/* Source, saved copies and the active representation are three different
            facts, so the active one is marked rather than merely selected. */}
        {model.variants.length > 0 ? (
          <div className="variants">
            <span className="group-label">Saved variants</span>
            {model.variants.map((variant) => (
              <button
                key={variant.bits}
                className="small variant"
                aria-pressed={model.active_variant === variant.bits}
                onClick={() => onActivate(variant.bits)}
                disabled={anyBusy || model.active_variant === variant.bits}
                title={variant.path}
              >
                Use {variant.bits}-bit
                {model.active_variant === variant.bits ? " · active" : ""}
              </button>
            ))}
            {model.active_variant !== null && (
              <button className="small" onClick={() => onActivate(null)} disabled={anyBusy}>
                Use original
              </button>
            )}
          </div>
        ) : (
          quant.supports_prequantize && (
            <p className="setting-help">
              No saved copy yet. Generation uses the source weights until one exists and is
              activated.
            </p>
          )
        )}

        {finishedHere && (
          <ActionNote state={{ status: "ok", message: finishedHere }} />
        )}

        {["convert", "activate"].map((kind) => (
          <ActionNote
            key={kind}
            state={stateOf(`${kind}:${model.key}`)}
            onDismiss={() => dismiss(`${kind}:${model.key}`)}
          />
        ))}
      </fieldset>
    </Modal>
  );
}
