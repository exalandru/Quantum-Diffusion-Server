import { useEffect, useRef, useState, type DragEvent } from "react";

import * as api from "../api";
import type { ModelCapabilities } from "../types";
import { AdvancedSettings, DEFAULT_ADVANCED, sizeOf, type Advanced } from "./AdvancedSettings";

export type Draft = {
  prompt: string;
  model: string;
  n: number;
  image: File | null;
  /** "WxH", or null for the model's default size. */
  size: string | null;
  /** null = the model's default step count. */
  steps: number | null;
  /** null = a random seed. */
  seed: number | null;
};

/**
 * Prompt, model, count, one optional reference image, and the advanced
 * parameters behind the gear — size, steps, seed.
 *
 * Anything left untouched stays the model's server-side default, which is where
 * those defaults already live: an auto aspect ratio sends no `size` at all.
 */
export function Composer({
  models,
  defaultModel,
  maxN,
  busy,
  error,
  onSubmit,
}: {
  /** Public names, as `/v1/models` lists them: `id` is what the API accepts. */
  models: { id: string; name: string }[];
  /** The server's default. May name a model by its internal id. */
  defaultModel: string;
  maxN: number;
  busy: boolean;
  error: string | null;
  onSubmit: (draft: Draft) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [chosen, setChosen] = useState<string | null>(null);
  const [n, setN] = useState(1);
  const [attachment, setAttachment] = useState<{ file: File; url: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [capabilities, setCapabilities] = useState<ModelCapabilities | null>(null);
  const [advanced, setAdvanced] = useState<Advanced>(DEFAULT_ADVANCED);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const advancedWrapper = useRef<HTMLDivElement>(null);

  const selected =
    chosen ??
    (models.some((entry) => entry.id === defaultModel) ? defaultModel : (models[0]?.id ?? ""));

  // Read per model, from `/v1/models/{id}`: `/v1/capabilities` is keyed by
  // internal id, and this page speaks public names throughout — the same ones
  // the generation record stores.
  useEffect(() => {
    if (!selected) return;
    let live = true;
    setCapabilities(null);
    void api
      .modelInfo(selected)
      .then((model) => {
        if (live) setCapabilities(model.mflux);
      })
      .catch(() => {
        // Left null: the drop zone stays disabled rather than offering an
        // attachment the model may not accept.
      });
    return () => {
      live = false;
    };
  }, [selected]);

  // One question, asked of the model that will run it: a reference image is only
  // meaningful if the model can edit or start from one.
  const acceptsImage = capabilities
    ? capabilities.supports_edit || capabilities.supports_image_to_image
    : false;

  // Switching to a model that takes no image drops the attachment rather than
  // keeping a thumbnail on screen and quietly generating without it.
  useEffect(() => {
    if (capabilities && !acceptsImage) clearAttachment();
  }, [capabilities, acceptsImage]);

  // The gear reads "on" only when something was actually chosen: an untouched
  // composer still generates at the model's own defaults.
  const advancedTouched =
    advanced.ratio !== "auto" || advanced.steps !== null || advanced.seed !== null;

  // While the popover is open, a press outside it or Escape closes it. mousedown
  // rather than click, so a selection dragged out of the panel is not a dismissal.
  useEffect(() => {
    if (!settingsOpen) return;
    function onPointer(event: MouseEvent) {
      if (!advancedWrapper.current?.contains(event.target as Node)) setSettingsOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setSettingsOpen(false);
    }
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [settingsOpen]);

  function attach(files: FileList | null) {
    const file = files?.[0];
    // One image: the server's edits contract takes one, so a second would be
    // dropped in silence. Re-dropping replaces.
    if (!file || !file.type.startsWith("image/")) return;
    setAttachment((current) => {
      if (current) URL.revokeObjectURL(current.url);
      return { file, url: URL.createObjectURL(file) };
    });
  }

  function clearAttachment() {
    setAttachment((current) => {
      if (current) URL.revokeObjectURL(current.url);
      return null;
    });
  }

  function submit() {
    if (!prompt.trim() || busy) return;
    onSubmit({
      prompt,
      model: selected,
      n,
      image: attachment?.file ?? null,
      size: sizeOf(advanced),
      steps: advanced.steps,
      seed: advanced.seed,
    });
    setPrompt("");
    clearAttachment();
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (acceptsImage) attach(event.dataTransfer.files);
  }

  return (
    <div className="pg-composer">
      <div
        className={dragging ? "pg-dropzone dragging" : "pg-dropzone"}
        onDragOver={(event) => {
          event.preventDefault();
          if (acceptsImage) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onPaste={(event) => {
          if (acceptsImage) attach(event.clipboardData.files);
        }}
      >
        {attachment ? (
          <div className="pg-attachment">
            <img src={attachment.url} alt="Reference" />
            <span className="pg-attachment-name">{attachment.file.name}</span>
            <button className="small" onClick={clearAttachment}>
              Remove
            </button>
          </div>
        ) : (
          <>
            <span className="pg-dropzone-label">Add Context (optional)</span>
            <button
              className="small"
              disabled={!acceptsImage}
              onClick={() => picker.current?.click()}
            >
              Attach image
            </button>
            <input
              ref={picker}
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => {
                attach(event.target.files);
                // Cleared so picking the same file again still fires `change`.
                event.target.value = "";
              }}
            />
            {capabilities && !acceptsImage && (
              <p className="note">{selected} supports neither editing nor image-to-image.</p>
            )}
          </>
        )}
      </div>

      <div className="pg-prompt-card">
        <textarea
          className="pg-textarea"
          rows={3}
          aria-label="Prompt"
          placeholder="Describe what's on your mind…"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            // Enter submits, Shift+Enter writes a newline: the convention of
            // every prompt box, and the reason this is a textarea at all.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <div className="pg-controls">
          <select
            aria-label="Model"
            value={selected}
            onChange={(event) => setChosen(event.target.value)}
          >
            {models.map(({ id, name }) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
          <div className="pg-stepper" role="group" aria-label="Images">
            <button
              className="small"
              aria-label="One less image"
              disabled={n <= 1}
              onClick={() => setN((value) => Math.max(1, value - 1))}
            >
              −
            </button>
            <span className="pg-count">{n}</span>
            <button
              className="small"
              aria-label="One more image"
              disabled={n >= maxN}
              onClick={() => setN((value) => Math.min(maxN, value + 1))}
            >
              +
            </button>
          </div>
          <div className="pg-advanced" ref={advancedWrapper}>
            <button
              type="button"
              className={advancedTouched ? "small pg-gear active" : "small pg-gear"}
              aria-label="Advanced settings"
              aria-expanded={settingsOpen}
              onClick={() => setSettingsOpen((open) => !open)}
            >
              <svg
                viewBox="0 0 24 24"
                width="14"
                height="14"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                aria-hidden="true"
              >
                <path d="M4 7h9M19 7h1M4 17h3M13 17h7" />
                <circle cx="15.5" cy="7" r="2.5" />
                <circle cx="9.5" cy="17" r="2.5" />
              </svg>
            </button>
            {settingsOpen && (
              <AdvancedSettings
                value={advanced}
                onChange={setAdvanced}
                capabilities={capabilities}
              />
            )}
          </div>
          <button className="primary" disabled={!prompt.trim() || busy} onClick={submit}>
            {busy ? "Sending…" : "Generate ↵"}
          </button>
        </div>
        {error && (
          <div className="notice notice-error" role="status">
            <strong>Not accepted.</strong> {error}
          </div>
        )}
      </div>
    </div>
  );
}
