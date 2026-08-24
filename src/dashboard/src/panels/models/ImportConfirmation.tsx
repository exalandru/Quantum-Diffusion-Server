import type { Capabilities, ImportVerdict } from "../../types";

export function ImportConfirmation({
  verdict,
  profile,
  name,
  apiName,
  capabilities,
  busy,
  onProfile,
  onName,
  onApiName,
  onConfirm,
  onCancel,
}: {
  verdict: ImportVerdict;
  profile: string;
  name: string;
  apiName: string;
  capabilities: Capabilities | null;
  busy: boolean;
  onProfile: (value: string) => void;
  onName: (value: string) => void;
  onApiName: (value: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <fieldset className="settings-group">
      <legend>Confirm import</legend>

      <div className="setting">
        <span className="setting-label">Folder</span>
        <code className="library-path">{verdict.path}</code>
      </div>

      <div className="setting">
        <span className="setting-label">Detected family</span>
        <div className="row">
          <span className="pill pill-accent">{verdict.family}</span>
          <span className="setting-help" style={{ margin: 0 }}>
            from <code>_class_name</code>: <code>{verdict.class_name}</code>
          </span>
        </div>
      </div>

      <div className="setting-pair">
        <div className="setting">
          <label className="setting-label" htmlFor="import-profile">
            Base profile
          </label>
          <select
            id="import-profile"
            aria-label="Base profile"
            value={profile}
            onChange={(event) => onProfile(event.target.value)}
          >
            <option value="">choose…</option>
            {verdict.profiles.map((candidate) => (
              <option key={candidate} value={candidate}>
                {candidate}
                {capabilities?.models[candidate]
                  ? ` - ${capabilities.models[candidate]!.default_steps} steps`
                  : ""}
              </option>
            ))}
          </select>
          <p className="setting-help">
            Supplies this model's generation defaults - steps, guidance, scheduler.
          </p>
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="import-name">
            Display name
          </label>
          <input
            id="import-name"
            type="text"
            value={name}
            onChange={(event) => onName(event.target.value)}
          />
          <p className="setting-help">How it appears in this list.</p>
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="import-api-name">
            API name
          </label>
          <input
            id="import-api-name"
            type="text"
            value={apiName}
            spellCheck={false}
            onChange={(event) => onApiName(event.target.value)}
          />
          {/* Three identities, and this is the machine-facing one. The internal
              id stays opaque and durable; the display name is for people. */}
          <p className="setting-help">
            What API requests send as <code>"model"</code>. Lowercase letters, digits,{" "}
            <code>.</code>, <code>_</code> or <code>-</code>, and unique across every model.
          </p>
        </div>
      </div>

      <div className="actions" style={{ marginTop: 14 }}>
        <button className="primary" onClick={onConfirm} disabled={!profile || !apiName || busy}>
          Register
        </button>
        <button onClick={onCancel}>Cancel</button>
        {!profile && (
          <span className="setting-help" style={{ margin: 0 }}>
            Choose the profile whose generation defaults this model should use.
          </span>
        )}
      </div>
    </fieldset>
  );
}
