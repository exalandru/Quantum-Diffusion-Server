import { useState } from "react";

import { ActionNote } from "../../actions";
import { Modal } from "../../modal";
import { Switch } from "./Switch";
import type { ModelCapabilities, ModelStatus } from "../../types";

/**
 * The per-model overrides, edited as a draft and applied in one write.
 *
 * A draft rather than a field-by-field write because these are typed values: a
 * write per keystroke would rewrite the configuration file five times while
 * someone types "1024" and hand the server a `1` to validate on the way.
 *
 * A dialog rather than a disclosure inside the row. Expanded in place, a form
 * this tall pushed every model below it off the screen, and the row it belonged
 * to became indistinguishable from the form's own header.
 *
 * Quantization is not here: it is two mechanisms with their own dialog, and a
 * third control called "Quantization" beside Steps was the ambiguity that made
 * a runtime setting look like it produced the saved copies listed one row up.
 *
 * Which controls are offered, and which are inert, is the backend's declaration —
 * `supports_guidance`, `supports_edit`. Nothing here decides what a model can do.
 */
export function ModelSettings({
  model,
  caps,
  override,
  busy,
  disabled,
  note,
  onDismissNote,
  onApply,
  onClose,
}: {
  model: ModelStatus;
  caps: ModelCapabilities | undefined;
  override: Record<string, any>;
  busy: boolean;
  disabled: boolean;
  note: Parameters<typeof ActionNote>[0]["state"];
  onDismissNote: () => void;
  onApply: (patch: Record<string, unknown>) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const value = (key: string) => (key in draft ? draft[key] : override[key]);
  const set = (key: string, next: unknown) => setDraft((previous) => ({ ...previous, [key]: next }));
  const dirty = Object.keys(draft).length > 0;

  const numeric = (key: string) => ({
    type: "number" as const,
    value: (value(key) ?? "") as string | number,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      set(key, event.target.value === "" ? null : Number(event.target.value)),
  });

  // Off because this model has no edit path, as opposed to off because it is
  // switched off: the second is ours to change, the first is not. A stale `true`
  // in the configuration stays operable, so it can be switched back off.
  const editingUnsupported = caps ? !caps.supports_edit && override.enable_edit !== true : false;

  return (
    <Modal
      title={`Model settings - ${model.display_name}`}
      subtitle={model.repo}
      onClose={onClose}
    >
      <div className="setting-pair">
        <div className="setting">
          <label className="setting-label" htmlFor={`steps-${model.key}`}>
            Steps
          </label>
          <input
            id={`steps-${model.key}`}
            aria-label={`Steps for ${model.key}`}
            min={1}
            placeholder={caps ? String(caps.default_steps) : "model default"}
            {...numeric("default_steps")}
          />
          <p className="setting-help">Denoising steps when a request names none.</p>
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor={`guidance-${model.key}`}>
            Guidance
          </label>
          <input
            id={`guidance-${model.key}`}
            aria-label={`Guidance for ${model.key}`}
            min={0}
            step={0.5}
            // Distilled model: the server rejects any value.
            disabled={caps ? !caps.supports_guidance : false}
            placeholder={
              caps?.supports_guidance === false
                ? `fixed ${caps.default_guidance ?? 0}`
                : caps
                  ? String(caps.default_guidance ?? "")
                  : "model default"
            }
            {...numeric("default_guidance")}
          />
          {caps?.supports_guidance === false && (
            <p className="setting-help">This model ignores guidance.</p>
          )}
        </div>
      </div>

      {/* A full-width row of its own, not a cell in the grid above. Sharing the
          narrow numeric columns broke "Expose the edits endpoint" over three
          lines beside a checkbox, and a wrapped label reads as a paragraph
          rather than as the name of the control next to it. */}
      <div className="setting">
        <span className="setting-label">Image editing</span>
        <div className="switch-row">
          <span className="switch-row-label">Expose the edits endpoint</span>
          <Switch
            checked={value("enable_edit") === true}
            label={`Expose the edits endpoint for ${model.key}`}
            disabled={editingUnsupported}
            onChange={(next) => set("enable_edit", next)}
          />
        </div>
        <p className="setting-help">
          {caps?.supports_edit === false
            ? "Not supported by this model: it has no image-editing path, so the endpoint would reject every request."
            : "Serves this model at /v1/images/edits as well as /v1/images/generations."}
        </p>
      </div>

      <div className="actions" style={{ marginTop: 16 }}>
        <button
          className="small primary"
          onClick={() => onApply(draft)}
          disabled={!dirty || busy || disabled}
        >
          {busy ? "Saving…" : "Apply"}
        </button>
        <button className="small" onClick={() => setDraft({})} disabled={!dirty || busy}>
          Reset
        </button>
        <span className="setting-help" style={{ margin: 0 }}>
          Applies when the generation server restarts.
        </span>
      </div>
      <ActionNote state={note} onDismiss={onDismissNote} />
    </Modal>
  );
}
