/**
 * The sidebar row's controls: what a locked session shows, and which actions a
 * row offers depending on whether this tab holds its token.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { playgroundSession } from "../test-fixtures";
import { SessionList } from "./SessionList";

// `unlocked` is spelled out rather than inferred from the default: the prop is
// `(id: string) => boolean`, and inferring `() => boolean` from `() => false`
// would reject the per-session predicates the tests below pass.
function renderList(
  sessions = [playgroundSession()],
  unlocked: (id: string) => boolean = () => false,
) {
  const spies = {
    onSelect: vi.fn(),
    onNew: vi.fn(),
    onRename: vi.fn(),
    onPassword: vi.fn(),
    onLock: vi.fn(),
    onDelete: vi.fn(),
  };
  render(
    <SessionList
      sessions={sessions}
      selected={null}
      unlocked={unlocked}
      {...spies}
    />,
  );
  return spies;
}

it("marks a locked session, and says when this tab has it open", () => {
  renderList(
    [
      playgroundSession({ id: "a", title: "closed", locked: true }),
      playgroundSession({ id: "b", title: "open", locked: true }),
    ],
    (id) => id === "b",
  );
  expect(screen.getByRole("img", { name: "Locked" })).toBeTruthy();
  expect(
    screen.getByRole("img", { name: "Locked, open in this tab" }),
  ).toBeTruthy();
  // An open session carries no glyph at all.
  renderList([playgroundSession({ id: "c", title: "plain" })]);
  expect(screen.queryByRole("img", { name: /plain/ })).toBeNull();
});

it("offers Lock only for a locked session this tab has unlocked", () => {
  renderList([playgroundSession({ id: "a", title: "closed", locked: true })]);
  expect(screen.queryByRole("button", { name: "Lock closed" })).toBeNull();
  expect(
    screen.getByRole("button", { name: "Change password of closed" }),
  ).toBeTruthy();

  renderList(
    [playgroundSession({ id: "b", title: "open", locked: true })],
    () => true,
  );
  expect(screen.getByRole("button", { name: "Lock open" })).toBeTruthy();

  renderList([playgroundSession({ id: "c", title: "plain" })]);
  expect(
    screen.getByRole("button", { name: "Set a password on plain" }),
  ).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Lock plain" })).toBeNull();
});

it("hands each action the session it belongs to", async () => {
  const user = userEvent.setup();
  const spies = renderList(
    [playgroundSession({ id: "s9", title: "fox", locked: true })],
    () => true,
  );
  await user.click(screen.getByRole("button", { name: "Rename fox" }));
  await user.click(
    screen.getByRole("button", { name: "Change password of fox" }),
  );
  await user.click(screen.getByRole("button", { name: "Lock fox" }));
  expect(spies.onRename).toHaveBeenCalledWith("s9");
  expect(spies.onPassword).toHaveBeenCalledWith("s9");
  expect(spies.onLock).toHaveBeenCalledWith("s9");

  vi.spyOn(window, "confirm").mockReturnValue(false);
  await user.click(screen.getByRole("button", { name: "Delete fox" }));
  expect(spies.onDelete).not.toHaveBeenCalled();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  await user.click(screen.getByRole("button", { name: "Delete fox" }));
  expect(spies.onDelete).toHaveBeenCalledWith("s9");
});
