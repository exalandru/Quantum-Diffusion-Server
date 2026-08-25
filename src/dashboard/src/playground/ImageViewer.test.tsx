/**
 * What the stripped-back viewer must still do.
 *
 * The redesign removed the dialog's head — no prompt as a title, no panel drawn
 * round the picture — and the three behaviours that head sat above are the ones
 * worth pinning, because they are exactly what a hand-rolled overlay would have
 * silently dropped: Escape, the backdrop press, and handing focus back to the
 * thumbnail that opened it. `Modal` still provides all three; `bare` hides
 * chrome, not behaviour, and these assertions are what says so.
 *
 * The fourth is the download. The URL carries the session unlock token as `?t=`,
 * so opening it as a navigation would write that token into the address bar and
 * the browser's history. It has to stay a `download` on an anchor.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, it, vi } from "vitest";

import { ImageViewer } from "./ImageViewer";

const URL_WITH_TOKEN = "/playground/images/f1.png?t=secret-unlock-token";
const PROMPT = "a lone lighthouse in a winter storm";

/** The viewer as a thumbnail opens it, so focus has somewhere to return to. */
function Opened({ onClose = () => {} }: { onClose?: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open the picture
      </button>
      {open && (
        <ImageViewer
          url={URL_WITH_TOKEN}
          caption={PROMPT}
          detail="seed 41 · 1440x800"
          onClose={() => {
            setOpen(false);
            onClose();
          }}
        />
      )}
    </>
  );
}

it("shows the picture and no prompt title", () => {
  render(<ImageViewer url={URL_WITH_TOKEN} caption={PROMPT} onClose={vi.fn()} />);
  const dialog = screen.getByRole("dialog");

  // The prompt names the dialog for a screen reader and is nowhere on screen:
  // it was the title, and a paragraph-long prompt took the top of the surface
  // away from the picture it describes.
  expect(dialog.getAttribute("aria-label")).toBe(PROMPT);
  expect(dialog.textContent).not.toContain(PROMPT);
  expect(dialog.querySelector(".modal-head")).toBeNull();
  expect(dialog.querySelector(".modal-title")).toBeNull();

  expect(screen.getByRole("img", { name: PROMPT }).getAttribute("src")).toBe(URL_WITH_TOKEN);
});

it("offers the file as a download, never as a navigation", () => {
  render(<ImageViewer url={URL_WITH_TOKEN} caption={PROMPT} onClose={vi.fn()} />);
  const link = screen.getByRole("link", { name: /Download/ });

  // The three properties together are the guarantee: an anchor, carrying
  // `download`, and not aimed at another tab. Any one of them alone would let
  // the `?t=` unlock token reach the address bar and the history.
  expect(link.tagName).toBe("A");
  expect(link.hasAttribute("download")).toBe(true);
  expect(link.getAttribute("href")).toBe(URL_WITH_TOKEN);
  expect(link.getAttribute("target")).toBeNull();
});

it("closes on its own cross, and hands focus back to what opened it", async () => {
  const user = userEvent.setup();
  const onClose = vi.fn();
  render(<Opened onClose={onClose} />);
  const opener = screen.getByRole("button", { name: "Open the picture" });

  await user.click(opener);
  await user.click(screen.getByRole("button", { name: "Close" }));
  expect(onClose).toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).toBeNull();
  // The round-trip. Dropping `Modal` would have cost this, and it is the one
  // behaviour a keyboard user notices immediately.
  expect(document.activeElement).toBe(opener);
});

it("still closes on Escape", async () => {
  const user = userEvent.setup();
  render(<Opened />);
  await user.click(screen.getByRole("button", { name: "Open the picture" }));
  expect(screen.getByRole("dialog")).toBeTruthy();

  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("still closes on a press of the backdrop", async () => {
  const user = userEvent.setup();
  const { container } = render(<Opened />);
  await user.click(screen.getByRole("button", { name: "Open the picture" }));

  const backdrop = container.querySelector(".modal-backdrop") as HTMLElement;
  // Press *and* release outside the surface, which is what RAC requires: a text
  // selection dragged out of the dialog must not dismiss it.
  await user.pointer([
    { target: backdrop, keys: "[MouseLeft>]" },
    { target: backdrop, keys: "[/MouseLeft]" },
  ]);
  expect(screen.queryByRole("dialog")).toBeNull();
});
