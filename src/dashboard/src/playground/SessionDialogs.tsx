/**
 * The three small forms a session row opens: rename, unlock, password.
 *
 * Each is a `Modal` around one form, the shape `PathPrompt` established. They
 * report the server's answer rather than validating ahead of it — the length
 * floor is mentioned as a hint, and the server still decides.
 */
import { useState, type FormEvent } from "react";

import { messageOf } from "../api";
import { Modal } from "../modal";

/** Mirrors the server's `credential.MIN_LENGTH`; a hint, not the rule. */
export const MIN_PASSWORD_LENGTH = 8;

/** Runs `action`, keeping the dialog open with the message when it fails. */
function useSubmit(action: () => Promise<void>) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, submit };
}

export function RenameDialog({
  title,
  onCancel,
  onRename,
}: {
  title: string | null;
  onCancel: () => void;
  /** Resolves once the server has the new name; rejects with its message. */
  onRename: (title: string | null) => Promise<void>;
}) {
  const [value, setValue] = useState(title ?? "");
  const { busy, error, submit } = useSubmit(() =>
    onRename(value.trim() || null),
  );

  return (
    <Modal title="Rename session" onClose={onCancel}>
      <form onSubmit={(event) => void submit(event)}>
        <p className="setting-help">
          Leave it empty to name the session after its first prompt.
        </p>
        <label htmlFor="session-title">Name</label>
        <input
          id="session-title"
          type="text"
          autoFocus
          maxLength={80}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        {error && (
          <p className="setting-error" role="alert">
            {error}
          </p>
        )}
        <div className="actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={busy}>
            Rename
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function UnlockDialog({
  title,
  onCancel,
  onUnlock,
}: {
  title: string | null;
  onCancel: () => void;
  /** Rejects with the server's message — wrong password, or "wait". */
  onUnlock: (password: string) => Promise<void>;
}) {
  const [password, setPassword] = useState("");
  const { busy, error, submit } = useSubmit(() => onUnlock(password));

  return (
    <Modal
      title="Unlock session"
      subtitle={title ?? undefined}
      onClose={onCancel}
    >
      <form onSubmit={(event) => void submit(event)}>
        <p className="setting-help">
          This session has a password. It stays unlocked in this tab until you
          lock it or close the tab.
        </p>
        <label htmlFor="session-password">Password</label>
        <input
          id="session-password"
          type="password"
          autoFocus
          autoComplete="off"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error && (
          <p className="setting-error" role="alert">
            {error}
          </p>
        )}
        <div className="actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="submit"
            className="primary"
            disabled={busy || !password}
          >
            Unlock
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function PasswordDialog({
  title,
  locked,
  onCancel,
  onSet,
  onRemove,
}: {
  title: string | null;
  /** Already has one: the form changes it, and offers to remove it. */
  locked: boolean;
  onCancel: () => void;
  onSet: (password: string) => Promise<void>;
  onRemove: () => Promise<void>;
}) {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [removing, setRemoving] = useState(false);
  const { busy, error, submit } = useSubmit(() => onSet(password));
  const [removeError, setRemoveError] = useState<string | null>(null);
  const mismatch = confirm.length > 0 && confirm !== password;
  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH;

  return (
    <Modal
      title={locked ? "Change password" : "Set a password"}
      subtitle={title ?? undefined}
      onClose={onCancel}
    >
      <form onSubmit={(event) => void submit(event)}>
        <p className="setting-help">
          A password locks this session's prompts and images. Anyone opening it
          — on this machine or another — is asked for it first.
          {locked && " Changing it signs every other tab out of this session."}
        </p>
        <label htmlFor="session-new-password">
          {locked ? "New password" : "Password"} (at least {MIN_PASSWORD_LENGTH}{" "}
          characters)
        </label>
        <input
          id="session-new-password"
          type="password"
          autoFocus
          autoComplete="new-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <label htmlFor="session-confirm-password">Confirm</label>
        <input
          id="session-confirm-password"
          type="password"
          autoComplete="new-password"
          value={confirm}
          onChange={(event) => setConfirm(event.target.value)}
        />
        {mismatch && <p className="setting-error">The two passwords differ.</p>}
        {(error || removeError) && (
          <p className="setting-error" role="alert">
            {error ?? removeError}
          </p>
        )}
        <div className="actions">
          {locked && (
            <button
              type="button"
              className="danger"
              disabled={busy || removing}
              onClick={() => {
                if (
                  !window.confirm(
                    "Remove the password? Anyone will be able to open this session.",
                  )
                )
                  return;
                setRemoving(true);
                setRemoveError(null);
                onRemove()
                  .catch((cause) => setRemoveError(messageOf(cause)))
                  .finally(() => setRemoving(false));
              }}
            >
              Remove password
            </button>
          )}
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="submit"
            className="primary"
            disabled={
              busy || removing || !password || tooShort || confirm !== password
            }
          >
            {locked ? "Change" : "Set password"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
