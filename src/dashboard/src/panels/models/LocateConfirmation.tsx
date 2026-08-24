import type { LocateVerdict } from "../../types";

/**
 * Confirming that a directory may be bound to a built-in catalogue entry.
 *
 * The distinction this surface exists to make: a cached repository *proves* which
 * repository it is, because huggingface_hub encodes it in the directory name. A
 * folder of compatible weights proves nothing of the sort, and saying "this is
 * FLUX.2-klein" about it would be QDS's assertion, not a fact. So the unproven
 * case is stated before the binding, not hidden behind a success message.
 */
export function LocateConfirmation({
  verdict,
  busy,
  onConfirm,
  onCancel,
}: {
  verdict: LocateVerdict;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <fieldset className="settings-group">
      <legend>Use this folder for {verdict.model}</legend>

      <div className="setting">
        <span className="setting-label">Folder</span>
        <code className="library-path">{verdict.path}</code>
      </div>

      <div className="setting">
        <span className="setting-label">Detected</span>
        <div className="row">
          <span className="pill pill-accent">{verdict.family}</span>
          <span className="setting-help" style={{ margin: 0 }}>
            from <code>_class_name</code>: <code>{verdict.class_name}</code>
          </span>
        </div>
      </div>

      {verdict.repo_verified ? (
        <p className="setting-help">
          This folder is a Hugging Face cache entry for <code>{verdict.detected_repo}</code> - the
          same repository the catalogue names, so its identity is confirmed.
        </p>
      ) : (
        <p className="caution">
          QDS can confirm this folder holds a compatible <strong>{verdict.family}</strong> model,
          but not that it is the exact repository the catalogue names
          {verdict.detected_repo ? (
            <>
              {" "}
              - the folder identifies itself as <code>{verdict.detected_repo}</code>
            </>
          ) : (
            " - it carries no Hugging Face cache metadata to check against"
          )}
          . Generation defaults will come from the catalogue entry regardless.
        </p>
      )}

      <div className="actions" style={{ marginTop: 14 }}>
        <button className="primary" onClick={onConfirm} disabled={busy}>
          {busy ? "Saving…" : "Use this folder"}
        </button>
        <button onClick={onCancel}>Cancel</button>
        <span className="setting-help" style={{ margin: 0 }}>
          Nothing is copied or moved.
        </span>
      </div>
    </fieldset>
  );
}
