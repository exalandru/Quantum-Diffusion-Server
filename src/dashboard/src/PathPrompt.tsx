/**
 * Asks for a directory path, in place of the native folder chooser.
 *
 * A web page cannot open one, and the alternative — a directory-browsing
 * endpoint — would mean the server enumerating the filesystem for anything that
 * can reach it. That is a real surface to buy a convenience, so the path is
 * typed instead.
 *
 * Nothing here validates it. The server inspects the directory and answers with
 * what it actually found — the family, the missing components, an unmounted
 * volume — and a guess made in the browser could only disagree with that.
 */
import { useState } from "react";

import { Modal } from "./modal";

export function PathPrompt({
  title,
  hint,
  placeholder,
  onCancel,
  onSubmit,
}: {
  title: string;
  hint: string;
  placeholder?: string;
  onCancel: () => void;
  onSubmit: (path: string) => void;
}) {
  const [value, setValue] = useState("");

  return (
    <Modal title={title} onClose={onCancel}>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const path = value.trim();
          if (path) onSubmit(path);
        }}
      >
        <p className="setting-help">{hint}</p>
        <label htmlFor="path-prompt">Full path</label>
        <input
          id="path-prompt"
          type="text"
          autoFocus
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          placeholder={placeholder ?? "/Volumes/Models/some-model"}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <div className="actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={!value.trim()}>
            Continue
          </button>
        </div>
      </form>
    </Modal>
  );
}
