/**
 * Why a project dialog's field is a field and not two elements on one line.
 *
 * What went wrong: every one of these four forms rendered a bare `<label>` next
 * to a bare `<input>`. A `<label>` is `display: inline` and an `<input>` is a
 * shrink-to-fit inline-block, so the two shared a line and the field took the
 * browser's default ~20 character width — "small and jammed against its Name
 * label", as reported. The helper sentence was rendered *before* the label,
 * where it reads as the dialog's subtitle rather than as help for the control.
 *
 * The fix is the sheet's own form vocabulary (`.setting` / `.setting-label` /
 * `.setting-help`), which the configuration panel and the model dialogs already
 * use, so these assertions ask the real cascade (`css: true`) what applies
 * rather than restating a rule.
 *
 * The real components are rendered rather than a markup fixture: the defect was
 * not in the stylesheet — those rules already existed and were correct — it was
 * that the dialogs did not use them. A fixture asserting the sheet would have
 * passed on the broken build.
 */

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import "../styles.css";
import {
  NewProjectDialog,
  PasswordDialog,
  RenameDialog,
  UnlockDialog,
} from "./SessionDialogs";

afterEach(cleanup);

/** The dialogs, each with the label of the field this file inspects. */
const DIALOGS: [string, () => void, string][] = [
  [
    "Rename project",
    () => render(<RenameDialog title="foxes" onCancel={vi.fn()} onRename={vi.fn()} />),
    "Name",
  ],
  [
    "New project",
    () => render(<NewProjectDialog onCancel={vi.fn()} onCreate={vi.fn()} />),
    "Name",
  ],
  [
    "Unlock project",
    () => render(<UnlockDialog title="foxes" onCancel={vi.fn()} onUnlock={vi.fn()} />),
    "Password",
  ],
  [
    "Set a password",
    () =>
      render(
        <PasswordDialog
          title="foxes"
          locked={false}
          onCancel={vi.fn()}
          onSet={vi.fn()}
          onRemove={vi.fn()}
        />,
      ),
    "Password",
  ],
];

describe.each(DIALOGS)("the %s dialog's field", (_name, mount, label) => {
  it("puts the label on its own line above a full-width control", () => {
    mount();
    const field = screen.getByLabelText(label);
    const row = field.closest(".setting");
    expect(row).not.toBeNull();

    // Block-level, so the input below it starts a new line. `inline` is what the
    // bare `<label>` was, and what put the two side by side.
    const caption = within(row as HTMLElement).getByText(label, { selector: "label" });
    expect(window.getComputedStyle(caption).display).not.toBe("inline");

    // The whole width of the dialog body, not the input's intrinsic 20 characters.
    const control = window.getComputedStyle(field);
    expect(control.width).toBe("100%");
    // Comfortable height: the sheet's control height, the same one the dialog's
    // buttons use.
    expect(control.minHeight).toBe("var(--control-h)");
  });

  it("reads its help after the control, not as a heading above the label", () => {
    mount();
    const row = screen.getByLabelText(label).closest(".setting") as HTMLElement;
    const help = row.querySelector(".setting-help");
    // The confirm field carries no help, and nothing here invents one for it.
    if (help === null) return;
    expect(
      help.compareDocumentPosition(screen.getByLabelText(label)) &
        Node.DOCUMENT_POSITION_PRECEDING,
    ).toBeTruthy();
  });
});
