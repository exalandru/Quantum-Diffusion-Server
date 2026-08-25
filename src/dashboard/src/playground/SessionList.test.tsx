/**
 * The rail's rows: what a locked project shows, which actions a row offers
 * depending on whether this tab holds its token, and what survives the collapse
 * to a 56px column of landmarks.
 */

import { cleanup, render, screen } from "@testing-library/react";
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
  collapsed = false,
) {
  const spies = {
    onSelect: vi.fn(),
    onNew: vi.fn(),
    onToggleCollapsed: vi.fn(),
    onRename: vi.fn(),
    onPassword: vi.fn(),
    onLock: vi.fn(),
    onDelete: vi.fn(),
  };
  render(
    <SessionList
      sessions={sessions}
      selected={null}
      collapsed={collapsed}
      // The queue's state, for the footer strip: not what any test here is
      // about, so the running case stands for both.
      paused={false}
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

it("names an unnamed project rather than showing a blank row", () => {
  // The server leaves `title` NULL until something names it — the user, or the
  // first prompt. A project created from the name dialog and then left empty is
  // now a legitimate row, so the rail must still say what it is.
  renderList([playgroundSession({ id: "s1", title: null })]);
  expect(screen.getByRole("button", { name: /Untitled project/ })).toBeTruthy();
});

it("collapses to landmarks that still say which project is which", async () => {
  const user = userEvent.setup();
  const spies = renderList(
    [
      playgroundSession({ id: "s1", title: "foxes", locked: true }),
      playgroundSession({ id: "s2", title: "lighthouses" }),
    ],
    () => false,
    true,
  );

  // No name fits at 56px, so the accessible name is the whole identification:
  // the visible letter is `aria-hidden`, and a rail that only rendered the
  // letter would be a column of initials to a screen reader.
  const foxes = screen.getByRole("button", { name: "foxes" });
  expect(foxes.textContent).toContain("F");
  expect(screen.getByRole("button", { name: "lighthouses" })).toBeTruthy();

  // Selecting is the one thing a landmark does. Renaming, locking and deleting
  // are not offered at this width, so they cannot be reached by mistake either.
  await user.click(foxes);
  expect(spies.onSelect).toHaveBeenCalledWith("s1");
  expect(screen.queryByRole("button", { name: "Rename foxes" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Delete foxes" })).toBeNull();

  // The way back out, and the way to a new project, stay on the rail.
  await user.click(screen.getByRole("button", { name: "Expand the project rail" }));
  expect(spies.onToggleCollapsed).toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "New project" }));
  expect(spies.onNew).toHaveBeenCalled();
});

it("draws the landmark from the first word that distinguishes the project", () => {
  // Titles are first prompts unless the user named the project, and prompts
  // start with an article far more often than not — so a mark taken from the
  // literal first character makes a column of identical "A"s.
  renderList(
    [
      playgroundSession({ id: "s1", title: "A lone lighthouse in a storm" }),
      playgroundSession({ id: "s2", title: "An Icelandic cabin at sunset" }),
      playgroundSession({ id: "s3", title: "The rooftop, in the rain" }),
    ],
    () => false,
    true,
  );
  expect(screen.getByRole("button", { name: /^A lone/ }).textContent).toContain("L");
  expect(screen.getByRole("button", { name: /^An Icelandic/ }).textContent).toContain("I");
  expect(screen.getByRole("button", { name: /^The rooftop/ }).textContent).toContain("R");

  // A one-word title that *is* the article keeps its own letter: there is no
  // second word to fall back to.
  cleanup();
  renderList([playgroundSession({ id: "s4", title: "The" })], () => false, true);
  expect(screen.getByRole("button", { name: "The" }).textContent).toContain("T");
});

it("gives each project a landmark of its own colour, keyed by id", () => {
  // The rail is sorted by `updated_at` and reorders on every generation, so a
  // hue derived from the row's position would identify the slot rather than the
  // project. Same two projects, opposite order, same two hues.
  const a = playgroundSession({ id: "aaa", title: "one" });
  const b = playgroundSession({ id: "bbb", title: "two" });
  renderList([a, b], () => false, true);
  const first = screen.getByRole("button", { name: "one" }).style.getPropertyValue("--pg-mark-hue");
  const second = screen.getByRole("button", { name: "two" }).style.getPropertyValue("--pg-mark-hue");
  expect(first).not.toBe(second);

  cleanup();
  renderList([b, a], () => false, true);
  expect(
    screen.getByRole("button", { name: "one" }).style.getPropertyValue("--pg-mark-hue"),
  ).toBe(first);
});

it("draws the project's cover when the payload carries one, in either state", () => {
  const withCover = playgroundSession({
    id: "s1",
    title: "foxes",
    cover: "/playground/images/abc.png/thumb",
  });

  for (const collapsed of [false, true]) {
    cleanup();
    renderList([withCover], () => false, collapsed);
    // Anchored: the expanded row also holds "Rename foxes" and "Delete foxes".
    const row = screen.getByRole("button", { name: /^foxes/ });
    const picture = row.querySelector("img");
    expect(picture?.getAttribute("src")).toBe("/playground/images/abc.png/thumb");
    // Decorative: the button already carries the project's name, and a screen
    // reader announcing it twice is worse than an unlabelled image.
    expect(picture?.getAttribute("alt")).toBe("");
    // The letter is the fallback, so it is *not* drawn behind the picture.
    expect(row.textContent).not.toContain("F");
  }
});

it("falls back to the letter landmark when there is no cover, locked included", () => {
  // `cover: null` means two different things and the rail must not tell them
  // apart: a project with no images, and a *locked* project whose cover the
  // server withholds because the list endpoint answers without an unlock token
  // and the cover URL is the capability that fetches the file.
  for (const collapsed of [false, true]) {
    cleanup();
    renderList(
      [
        playgroundSession({ id: "s1", title: "empty", cover: null }),
        playgroundSession({ id: "s2", title: "sealed", locked: true, cover: null }),
      ],
      () => false,
      collapsed,
    );
    for (const name of ["empty", "sealed"]) {
      const row = screen.getByRole("button", { name: new RegExp(`^(Locked )?${name}`) });
      expect(row.querySelector("img")).toBeNull();
      expect(row.textContent).toContain(name[0]!.toUpperCase());
      // The hue rides on whichever element paints the tile: the landmark button
      // itself when collapsed, the row's leading tile when expanded.
      const tile = collapsed
        ? row
          : (row.querySelector(".pg-session-thumb") as HTMLElement);
      expect(tile.style.getPropertyValue("--pg-mark-hue")).not.toBe("");
    }
  }
});
