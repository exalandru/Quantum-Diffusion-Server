/**
 * The four small forms a project opens: create, rename, unlock, password.
 *
 * Each is a `Modal` around one form, the shape `PathPrompt` established. They
 * report the server's answer rather than validating ahead of it — the length
 * floor is mentioned as a hint, and the server still decides.
 *
 * The vocabulary here is the interface's, not the API's: these say "project"
 * while every route, payload and MCP tool underneath keeps saying "session".
 * That split is deliberate — renaming a public route for a UI word would be a
 * contract change with no caller asking for it.
 */
import { useState, type FormEvent, type ReactNode } from "react";

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

/**
 * One labelled control, laid out: label above, control full width beneath it,
 * help under the control.
 *
 * These four forms rendered a bare `<label>` next to a bare `<input>`, which is
 * not a layout at all — a `<label>` is inline and an `<input>` is a shrink-to-fit
 * inline-block, so the two sat on one line with the field at the browser's
 * default ~20 character width, jammed against its own label. And the helper
 * sentence sat *above* the label, where the eye reads it as the dialog's
 * subtitle rather than as help for the field under it.
 *
 * `.setting` / `.setting-label` / `.setting-help` are the sheet's existing form
 * vocabulary — the configuration panel, the model dialogs and the import forms
 * all use it, and `.setting input { width: 100% }` is where the full width comes
 * from. So this is one component adopting a convention rather than a fifth way
 * to draw a field, and it is a component rather than four copies of the same
 * three elements because the defect was that all four dialogs drifted the same
 * way once.
 */
function Field({
  id,
  label,
  help,
  children,
}: {
  id: string;
  label: ReactNode;
  /** Under the control, where it reads as help. Optional. */
  help?: ReactNode;
  /** The control itself. Its `id` must be the one above. */
  children: ReactNode;
}) {
  return (
    <div className="setting">
      <label className="setting-label" htmlFor={id}>
        {label}
      </label>
      {children}
      {help && <p className="setting-help">{help}</p>}
    </div>
  );
}

/**
 * Name a project, and create it.
 *
 * The project does not exist yet when this opens: `onCreate` is what creates it.
 * Cancelling therefore leaves nothing behind, which is the one thing that had to
 * be true of creating a project up front — a record the server holds and the
 * rail cannot show would be worse than no record at all.
 */
export function NewProjectDialog({
  onCancel,
  onCreate,
}: {
  onCancel: () => void;
  /** Resolves once the server holds a project with this name. */
  onCreate: (title: string) => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const { busy, error, submit } = useSubmit(() => onCreate(value.trim()));

  return (
    <Modal title="New project" onClose={onCancel}>
      <form onSubmit={(event) => void submit(event)}>
        <Field
          id="project-title"
          label="Name"
          help="A project holds prompts and the images they produced. You can rename it later."
        >
          <input
            id="project-title"
            type="text"
            autoFocus
            maxLength={80}
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
        </Field>
        {error && (
          <p className="setting-error" role="alert">
            {error}
          </p>
        )}
        <div className="actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={busy || !value.trim()}>
            Create
          </button>
        </div>
      </form>
    </Modal>
  );
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
    <Modal title="Rename project" onClose={onCancel}>
      <form onSubmit={(event) => void submit(event)}>
        <Field
          id="session-title"
          label="Name"
          help="Leave it empty to name the project after its first prompt."
        >
          <input
            id="session-title"
            type="text"
            autoFocus
            maxLength={80}
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
        </Field>
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
      title="Unlock project"
      subtitle={title ?? undefined}
      onClose={onCancel}
    >
      <form onSubmit={(event) => void submit(event)}>
        <Field
          id="session-password"
          label="Password"
          help="This project has a password. It stays unlocked in this tab until you lock it or close the tab."
        >
          <input
            id="session-password"
            type="password"
            autoFocus
            autoComplete="off"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </Field>
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
        <Field
          id="session-new-password"
          label={locked ? "New password" : "Password"}
          help={
            <>
              At least {MIN_PASSWORD_LENGTH} characters. A password locks this
              project's prompts and images: anyone opening it — on this machine or
              another — is asked for it first.
              {locked && " Changing it signs every other tab out of this project."}
            </>
          }
        >
          <input
            id="session-new-password"
            type="password"
            autoFocus
            autoComplete="new-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </Field>
        <Field id="session-confirm-password" label="Confirm">
          <input
            id="session-confirm-password"
            type="password"
            autoComplete="new-password"
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
          />
        </Field>
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
                    "Remove the password? Anyone will be able to open this project.",
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
