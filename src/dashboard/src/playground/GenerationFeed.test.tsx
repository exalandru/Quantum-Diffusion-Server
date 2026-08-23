/**
 * What one feed entry is, now that a generation is no longer one entry.
 *
 * Refining or varying an image submits another generation of the same lineage,
 * and the feed has to draw that lineage as a single entry — one prompt, the
 * images side by side, as if the whole set had been asked for at once. The two
 * things that can silently break it are the folding (one entry per `groupId`,
 * not per generation) and which member the entry treats as its root, because the
 * root is what every later refine and variation reuses as its request.
 *
 * Rendered rather than unit-tested: the grouping is only observable through what
 * the entry shows and through the payload the toolbar hands back.
 */

import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { type Mock, expect, it, vi } from "vitest";

import type { PlaygroundGeneration, Progress, Upscaler } from "../types";
import { GenerationFeed } from "./GenerationFeed";

const IDLE: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  preview_seq: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

const UPSCALERS: Upscaler[] = [
  {
    id: "realesrgan-x4plus",
    name: "Real-ESRGAN ×4 (photo)",
    scales: [2, 4],
    downloaded: true,
    sizeMb: 33.5,
    license: "BSD-3-Clause",
  },
  {
    id: "realesrgan-x4plus-anime",
    name: "Real-ESRGAN ×4 (illustration)",
    scales: [2, 4],
    downloaded: false,
    sizeMb: 9,
    license: "BSD-3-Clause",
  },
];

// jsdom has no layout engine and so no `scrollIntoView`. The feed calls it to
// keep new activity in view, which is not what these tests observe.
Element.prototype.scrollIntoView = () => {};

/**
 * The item at `index`, asserted present.
 *
 * `noUncheckedIndexedAccess` is right in general: an array index really can
 * miss. Each use below has already established the length it depends on, so
 * this states that dependency once and fails at the index — naming the missing
 * position — rather than inside a matcher three lines later.
 */
function nth<T>(items: T[], index: number): T {
  const item = items[index];
  expect(item).toBeDefined();
  return item as T;
}

function generation(patch: Partial<PlaygroundGeneration> = {}): PlaygroundGeneration {
  const id = patch.id ?? "g1";
  return {
    id,
    sessionId: "s1",
    groupId: id,
    prompt: "a fox",
    rewrittenPrompt: null,
    rewriteError: null,
    negativePrompt: null,
    model: "qwen-image-2512",
    kind: "txt2img",
    n: 1,
    size: "512x288",
    steps: 6,
    seeds: [41],
    contextImage: null,
    status: "completed",
    error: null,
    images: [{ url: `/playground/images/${id}.png`, seed: 41 }],
    createdAt: 1,
    startedAt: 1,
    finishedAt: 2,
    ...patch,
  };
}

/** The callbacks the feed reports interaction through. */
type Spies = {
  onCancel: Mock;
  onRefine: Mock;
  onVariation: Mock;
  onUpscale: Mock;
  onDeleteImage: Mock;
  onDeleteGroup: Mock;
};

function element(
  generations: PlaygroundGeneration[],
  spies: Spies,
  // The feed is handed the lookup rather than owning it, so the default here is
  // the identity: these tests observe grouping, not naming.
  nameOf: (id: string) => string = (id) => id,
  // Only the preview test cares: everything else observes a feed at rest.
  progress: Progress = IDLE,
  // Only the paused tests care.
  paused = false,
  // Empty is the fail-closed case, which one test exercises on purpose.
  upscalers: Upscaler[] = UPSCALERS,
) {
  return (
    <GenerationFeed
      generations={generations}
      progress={progress}
      onCancel={spies.onCancel}
      cancelling={null}
      busy={false}
      onRefine={spies.onRefine}
      onVariation={spies.onVariation}
      onUpscale={spies.onUpscale}
      upscalers={upscalers}
      onDeleteImage={spies.onDeleteImage}
      onDeleteGroup={spies.onDeleteGroup}
      paused={paused}
      nameOf={nameOf}
    />
  );
}

function feed(
  generations: PlaygroundGeneration[],
  spies: Spies,
  nameOf?: (id: string) => string,
) {
  return render(element(generations, spies, nameOf));
}

function handlers(): Spies {
  return {
    onCancel: vi.fn(),
    onRefine: vi.fn(),
    onVariation: vi.fn(),
    onUpscale: vi.fn(),
    onDeleteImage: vi.fn(),
    onDeleteGroup: vi.fn(),
  };
}

it("draws one lineage as one entry, whatever it was asked in", () => {
  const root = generation({ id: "root", prompt: "a fox" });
  const joined = generation({
    id: "joined",
    groupId: "root",
    // The member's own record differs from the root's on every field the entry
    // shows: if the entry read the member, these are what would appear.
    prompt: "a fox refined",
    rewrittenPrompt: null,
    rewriteError: null,
    seeds: [7],
    contextImage: "/playground/images/ctx-joined.png",
  });
  const unrelated = generation({ id: "other", prompt: "a bear" });

  feed([root, joined, unrelated], handlers(), (id) =>
    id === "qwen-image-2512" ? "Qwen" : id,
  );

  const entries = screen.getAllByRole("article");
  expect(entries).toHaveLength(2);
  // Two images in the first entry, one prompt, and the count the meta line
  // claims is the whole group's.
  expect(within(nth(entries, 0)).getAllByRole("toolbar")).toHaveLength(2);
  expect(within(nth(entries, 0)).getByText("a fox")).toBeTruthy();
  expect(within(nth(entries, 0)).queryByText("a fox refined")).toBeNull();
  expect(within(nth(entries, 0)).getByText(/2 images/)).toBeTruthy();
  // The member's reference image is not the entry's: only the original request's
  // is, and this group had none.
  expect(within(nth(entries, 0)).queryByAltText("Reference image")).toBeNull();
  expect(within(nth(entries, 1)).getByText("a bear")).toBeTruthy();
  // The meta line names the model the way a person reads it; the API identifier
  // is what the record stores and not what this line is for.
  const meta = nth(entries, 0).querySelector(".pg-meta");
  expect(meta?.textContent).toContain("Qwen");
  expect(meta?.textContent).not.toContain("qwen-image-2512");
});

it("hands the group's root to both generating actions, and the clicked image to refine", async () => {
  const root = generation({ id: "root" });
  const joined = generation({ id: "joined", groupId: "root", prompt: "drifted" });
  const spies = handlers();
  feed([root, joined], spies);

  const toolbars = screen.getAllByRole("toolbar");
  // The second image — the one produced by the refine, not the original.
  await userEvent.click(within(nth(toolbars, 1)).getByRole("button", { name: "Refine" }));
  await userEvent.click(within(nth(toolbars, 1)).getByRole("button", { name: "New variation" }));

  expect(spies.onRefine).toHaveBeenCalledWith(root, {
    url: "/playground/images/joined.png",
    seed: 41,
  });
  // Variation restarts from the original request and is told nothing about the
  // image it was clicked on.
  expect(spies.onVariation).toHaveBeenCalledWith(root);
});

it("keeps the destructive action last, behind a confirmation", async () => {
  const spies = handlers();
  feed([generation({ id: "root" })], spies);

  const tools = within(screen.getByRole("toolbar")).getAllByRole("button");
  expect(tools.map((button) => button.getAttribute("aria-label"))).toEqual([
    "Refine",
    "New variation",
    "Upscale",
    "Delete image",
  ]);
  // Nothing here says its name in a native tooltip: the visual one is CSS.
  expect(tools.every((button) => button.getAttribute("title") === null)).toBe(true);

  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
  await userEvent.click(nth(tools, 3));
  expect(spies.onDeleteImage).not.toHaveBeenCalled();
  confirm.mockReturnValue(true);
  await userEvent.click(nth(tools, 3));
  expect(spies.onDeleteImage).toHaveBeenCalledWith("/playground/images/root.png");
  confirm.mockRestore();
});

it("keeps every member's own status inside the shared entry", () => {
  const root = generation({ id: "root" });
  const failed = generation({
    id: "failed",
    groupId: "root",
    status: "failed",
    error: "Out of memory",
    images: [],
  });
  const queued = generation({ id: "queued", groupId: "root", status: "queued", images: [] });

  feed([root, failed, queued], handlers());

  const entry = screen.getByRole("article");
  expect(within(entry).getByText(/Out of memory/)).toBeTruthy();
  expect(within(entry).getByText("Queued…")).toBeTruthy();
  // One image, three requests: the meta line counts what was asked for.
  expect(within(entry).getAllByRole("toolbar")).toHaveLength(1);
  expect(within(entry).getByText(/3 images/)).toBeTruthy();
});

it("scrolls down when an entry appears, and never when one grows or goes", () => {
  const scroll = vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
  try {
    const spies = handlers();
    const root = generation({ id: "root" });
    const second = generation({ id: "second", prompt: "a bear" });
    const { rerender } = feed([root, second], spies);
    const onMount = scroll.mock.calls.length;

    // A refine or a variation joining the group — the click, then its image
    // landing. Neither is a new entry, so the feed must not move: the user may
    // have scrolled up to the picture they are working on.
    const joined = generation({ id: "joined", groupId: "root", images: [], status: "running" });
    rerender(element([root, joined, second], spies));
    expect(scroll.mock.calls.length).toBe(onMount);
    const landed = { ...joined, status: "completed" as const, images: [{ url: "/j.png", seed: 8 }] };
    rerender(element([root, landed, second], spies));
    expect(scroll.mock.calls.length).toBe(onMount);

    // Deleting the last image of an entry dissolves it, so an entry *leaves* the
    // feed. That is not something to scroll to either.
    rerender(element([root, landed], spies));
    expect(scroll.mock.calls.length).toBe(onMount);

    // A new prompt from the composer: a new entry, at the bottom, where the
    // composer already is.
    rerender(element([root, landed, generation({ id: "fresh", prompt: "a cat" })], spies));
    expect(scroll.mock.calls.length).toBe(onMount + 1);

    // Another session's transcript: land on its newest entry.
    rerender(element([generation({ id: "elsewhere", sessionId: "s2" })], spies));
    expect(scroll.mock.calls.length).toBe(onMount + 2);
  } finally {
    scroll.mockRestore();
  }
});

it("shows a placeholder, then crossfades one preview frame into the next", () => {
  vi.useFakeTimers();
  try {
    const spies = handlers();
    const running = generation({ id: "run", status: "running", images: [], n: 1 });
    const generating = (preview_seq: number): Progress => ({
      ...IDLE,
      state: "generating",
      step: preview_seq || 1,
      total: 6,
      preview_seq,
    });

    const { container, rerender } = render(
      element([running], spies, undefined, generating(0)),
    );
    // The box is there before the first frame is: it is what stops the feed from
    // jumping when one arrives, and it carries the request's aspect ratio.
    const box = container.querySelector(".pg-preview") as HTMLElement;
    expect(box).not.toBeNull();
    expect(box.style.aspectRatio).toBe("512 / 288");
    expect(box.querySelectorAll("img")).toHaveLength(0);
    // The "still working" glow is there from the start too, and it is the last
    // child so it lies over the frames rather than under them.
    expect(box.querySelector(".pg-preview-glow")).not.toBeNull();
    expect(box.lastElementChild?.className).toBe("pg-preview-glow");

    rerender(element([running], spies, undefined, generating(2)));
    const first = box.querySelectorAll("img");
    expect(first).toHaveLength(1);
    expect(first[0]?.getAttribute("src")).toBe("/playground/api/preview?v=2");

    // The next frame stacks on top; both are on screen for the length of the fade.
    rerender(element([running], spies, undefined, generating(4)));
    expect([...box.querySelectorAll("img")].map((img) => img.getAttribute("src"))).toEqual([
      "/playground/api/preview?v=2",
      "/playground/api/preview?v=4",
    ]);

    // jsdom never loads an image, so the browser's own event is played by hand.
    const fresh = box.querySelectorAll("img")[1] as HTMLImageElement;
    fireEvent.load(fresh);
    expect(fresh.className).toBe("on");
    act(() => void vi.advanceTimersByTime(500));
    expect([...box.querySelectorAll("img")].map((img) => img.getAttribute("src"))).toEqual([
      "/playground/api/preview?v=4",
    ]);

    // The run ended between the SSE frame and the fetch: the slot 404s, and the
    // dead frame goes without taking the visible one with it.
    rerender(element([running], spies, undefined, generating(6)));
    fireEvent.error(box.querySelectorAll("img")[1] as HTMLImageElement);
    expect([...box.querySelectorAll("img")].map((img) => img.getAttribute("src"))).toEqual([
      "/playground/api/preview?v=4",
    ]);

    // Somebody else's `/v1` job takes the engine, or the next image of the same
    // request starts: no frame belongs to what is running, so back to the box.
    rerender(element([running], spies, undefined, generating(0)));
    expect(box.querySelectorAll("img")).toHaveLength(0);
  } finally {
    vi.useRealTimers();
  }
});

it("gives the preview the grid cell the running image will land in", () => {
  const spies = handlers();
  const done = generation({ id: "root", images: [{ url: "/playground/images/root.png", seed: 41 }] });
  // What a refine or a variation is: another member of the same group, running,
  // with no image of its own yet.
  const refining = generation({ id: "refine", groupId: "root", status: "running", images: [] });
  const { container } = render(
    element([done, refining], spies, undefined, {
      ...IDLE,
      state: "generating",
      step: 2,
      total: 6,
      preview_seq: 2,
    }),
  );

  // Inside the grid and last, so it reads as the next image rather than as a
  // second widget below the finished ones.
  const cells = [...(container.querySelector(".pg-images")?.children ?? [])];
  expect(cells.map((cell) => cell.className)).toEqual(["pg-image-cell", "pg-preview"]);
  // And nowhere else: the running block below keeps the bar and the status only.
  expect(container.querySelector(".pg-running .pg-preview")).toBeNull();
});

// ── Negative prompts ───────────────────────────────────────────────────────

it("shows the negative prompt of the request the entry is made of", () => {
  // The entry shows the *root's* request, so a member disagreeing must not be
  // what is read: it is the root's settings a refine of this group reuses.
  feed(
    [
      generation({ id: "g1", negativePrompt: "blurry, watermark" }),
      generation({ id: "g2", groupId: "g1", negativePrompt: "something else" }),
    ],
    handlers(),
  );

  expect(screen.getByText(/blurry, watermark/)).toBeTruthy();
  expect(screen.queryByText(/something else/)).toBeNull();
});

it("says nothing about a negative prompt when none was sent", () => {
  const { container } = feed([generation()], handlers());
  expect(container.querySelector(".pg-prompt-negative")).toBeNull();
});

// ── Deleting a whole entry ─────────────────────────────────────────────────

it("deletes the group, not the generation that happens to be the root", () => {
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  const spies = handlers();
  // A lineage whose root's own id differs from the group's, so passing the
  // wrong one is visible rather than accidentally right.
  feed([generation({ id: "g7", groupId: "lineage-1" })], spies);

  screen.getByRole("button", { name: "Delete entry: a fox" }).click();

  expect(spies.onDeleteGroup).toHaveBeenCalledWith("lineage-1");
  confirm.mockRestore();
});

it("asks first, and does nothing when the answer is no", () => {
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
  const spies = handlers();
  feed([generation()], spies);

  screen.getByRole("button", { name: "Delete entry: a fox" }).click();

  expect(confirm).toHaveBeenCalled();
  expect(spies.onDeleteGroup).not.toHaveBeenCalled();
  confirm.mockRestore();
});

it("counts the whole lineage's images in what it asks", () => {
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
  feed(
    [
      generation({ id: "g1" }),
      generation({ id: "g2", groupId: "g1", images: [{ url: "/b.png", seed: 2 }] }),
    ],
    handlers(),
  );

  screen.getByRole("button", { name: "Delete entry: a fox" }).click();

  expect(confirm).toHaveBeenCalledWith("Delete this entry and its 2 images?");
  confirm.mockRestore();
});

// ── A held queue ───────────────────────────────────────────────────────────

it("says a queued generation is held rather than waiting its turn", () => {
  const spies = handlers();
  const queued = [generation({ status: "queued", images: [] })];

  const { rerender } = render(element(queued, spies, undefined, IDLE, false));
  expect(screen.getByText("Queued…")).toBeTruthy();

  rerender(element(queued, spies, undefined, IDLE, true));
  expect(screen.queryByText("Queued…")).toBeNull();
  expect(screen.getByText(/the queue is paused/i)).toBeTruthy();
});

// ── The preview blur ───────────────────────────────────────────────────────

it("eases the preview blur off as the run progresses", () => {
  const running = [generation({ status: "running", images: [] })];
  const at = (step: number): Progress => ({
    ...IDLE,
    state: "generating",
    step,
    total: 30,
    preview_seq: 1,
  });

  const spies = handlers();
  const { container, rerender } = render(element(running, spies, undefined, at(1)));
  const box = () => container.querySelector(".pg-preview") as HTMLElement;
  const blurAt = () => Number.parseFloat(box().style.getPropertyValue("--pg-preview-blur"));

  const early = blurAt();
  rerender(element(running, spies, undefined, at(15)));
  const middle = blurAt();
  rerender(element(running, spies, undefined, at(30)));
  const late = blurAt();

  expect(early).toBeGreaterThan(middle);
  expect(middle).toBeGreaterThan(late);
  // Nearly sharp by the end of a 30-step run — the denoiser has done the job
  // the blur was standing in for.
  expect(late).toBeLessThan(3);
});

it("keeps a short run blurrier at its last step than a long one", () => {
  const running = [generation({ status: "running", images: [] })];
  const done = (total: number): Progress => ({
    ...IDLE,
    state: "generating",
    step: total,
    total,
    preview_seq: 1,
  });
  const spies = handlers();

  const { container, rerender } = render(element(running, spies, undefined, done(8)));
  const blurAt = () =>
    Number.parseFloat(
      (container.querySelector(".pg-preview") as HTMLElement).style.getPropertyValue(
        "--pg-preview-blur",
      ),
    );

  const short = blurAt();
  rerender(element(running, spies, undefined, done(50)));
  expect(short).toBeGreaterThan(blurAt());
});

it("holds the full blur while the step count is somebody else's", () => {
  // Weights loading, or an external `/v1` client holding the engine: the same
  // guard that blanks `seq` must blank the blur's denominator.
  const spies = handlers();
  const { container } = render(
    element([generation({ status: "running", images: [] })], spies, undefined, {
      ...IDLE,
      state: "loading",
      step: 20,
      total: 30,
    }),
  );

  const box = container.querySelector(".pg-preview") as HTMLElement;
  expect(Number.parseFloat(box.style.getPropertyValue("--pg-preview-blur"))).toBe(20);
});


// ── Upscaling ──────────────────────────────────────────────────────────────

it("takes a factor and a model, and hands back the image it was opened on", async () => {
  const spies = handlers();
  const root = generation({ id: "root" });
  feed([root], spies);

  const toolbar = screen.getByRole("toolbar");
  await userEvent.click(within(toolbar).getByRole("button", { name: "Upscale" }));

  const panel = screen.getByRole("dialog", { name: "Upscale options" });
  // The panel is a sibling of the toolbar, not a member of it: a toolbar holds
  // peer controls, and this keeps its four buttons four while the panel is open.
  expect(within(toolbar).getAllByRole("button")).toHaveLength(4);

  await userEvent.click(within(panel).getByRole("button", { name: "×4" }));
  await userEvent.click(
    within(panel).getByRole("button", { name: "Real-ESRGAN ×4 (illustration)" }),
  );
  await userEvent.click(within(panel).getByRole("button", { name: "Upscale" }));

  expect(spies.onUpscale).toHaveBeenCalledWith(
    root,
    { url: "/playground/images/root.png", seed: 41 },
    { model: "realesrgan-x4plus-anime", scale: 4 },
  );
  // And it closes behind itself.
  expect(screen.queryByRole("dialog", { name: "Upscale options" })).toBeNull();
});

it("says when the first use will cost a download, and only then", async () => {
  feed([generation({ id: "root" })], handlers());
  await userEvent.click(screen.getByRole("button", { name: "Upscale" }));
  const panel = screen.getByRole("dialog", { name: "Upscale options" });

  // The default entry is already downloaded: nothing to warn about.
  expect(within(panel).queryByText(/First use downloads/)).toBeNull();
  await userEvent.click(
    within(panel).getByRole("button", { name: "Real-ESRGAN ×4 (illustration)" }),
  );
  expect(within(panel).getByText("First use downloads 9 MB.")).toBeTruthy();
});

it("stays inert when the server offered no upscaler", () => {
  render(element([generation({ id: "root" })], handlers(), undefined, IDLE, false, []));
  const button = within(screen.getByRole("toolbar")).getAllByRole("button")[2] as HTMLButtonElement;
  expect(button.getAttribute("aria-label")).toBe("Upscale unavailable");
  expect(button.disabled).toBe(true);
});

it("labels an upscaled image, which the entry's own heading cannot", () => {
  const root = generation({ id: "root", size: "512x288" });
  const upscaled = generation({
    id: "up",
    groupId: "root",
    kind: "upscale",
    model: "realesrgan-x4plus",
    size: "2048x1152",
    images: [{ url: "/playground/images/up.png", seed: 42 }],
  });
  feed([root, upscaled], handlers());

  // The heading still reads the root's request, which is right -- the upscale
  // joined that lineage. The image itself has to say what it is.
  expect(screen.getByText("Upscaled · 2048x1152")).toBeTruthy();
});

it("counts tiles, not steps, while an upscale runs", () => {
  const root = generation({ id: "root" });
  const running = generation({
    id: "up",
    groupId: "root",
    kind: "upscale",
    status: "running",
    images: [],
  });
  render(
    element([root, running], handlers(), undefined, {
      ...IDLE,
      state: "upscaling",
      step: 3,
      total: 9,
    }),
  );
  expect(screen.getByText(/Upscaling - tile 3 of 9/)).toBeTruthy();
});

it("closes the panel from the button that opened it", async () => {
  feed([generation({ id: "root" })], handlers());
  const open = screen.getByRole("button", { name: "Upscale" });

  await userEvent.click(open);
  expect(screen.queryByRole("dialog", { name: "Upscale options" })).not.toBeNull();
  expect(open.getAttribute("aria-expanded")).toBe("true");

  // Without the trigger being exempt from "pressed outside", `mousedown`
  // closes and the button's own `click` reopens: a toggle that never toggles
  // off, while `aria-expanded` announces one that does.
  await userEvent.click(open);
  expect(screen.queryByRole("dialog", { name: "Upscale options" })).toBeNull();
});

it("still closes on Escape and on a press outside", async () => {
  feed([generation({ id: "root" })], handlers());
  await userEvent.click(screen.getByRole("button", { name: "Upscale" }));
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("dialog", { name: "Upscale options" })).toBeNull();

  await userEvent.click(screen.getByRole("button", { name: "Upscale" }));
  fireEvent.mouseDown(document.body);
  expect(screen.queryByRole("dialog", { name: "Upscale options" })).toBeNull();
});


// ── The enhanced prompt in the feed ────────────────────────────────────────

it("titles an entry with what the user typed, not with the rewrite", () => {
  feed(
    [
      generation({
        prompt: "un chat sur un toit",
        rewrittenPrompt: "A ginger cat on weathered terracotta tiles at dusk",
      }),
    ],
    handlers(),
  );
  // Nobody is shown words they did not write as though they had written them.
  expect(screen.getByText("un chat sur un toit")).toBeTruthy();
  expect(screen.getByText(/Enhanced prompt/i)).toBeTruthy();
  expect(screen.getByText(/A ginger cat on weathered terracotta tiles/)).toBeTruthy();
});

it("says a failed enhance still produced the image", () => {
  feed(
    [
      generation({
        prompt: "un chat sur un toit",
        rewriteError: "the model answered the user instead of rewriting a prompt",
      }),
    ],
    handlers(),
  );
  // Distinguishable from "no rewrite was asked for", which shows nothing.
  expect(screen.getByText(/Enhancing failed/i)).toBeTruthy();
  expect(screen.getByText(/generated from your prompt/i)).toBeTruthy();
});

it("shows nothing about rewriting when none was asked for", () => {
  feed([generation({ prompt: "a cat" })], handlers());
  expect(screen.queryByText(/Enhanced prompt/i)).toBeNull();
  expect(screen.queryByText(/Enhancing failed/i)).toBeNull();
});

it("hands an enhanced prompt back to the composer", async () => {
  const onUsePrompt = vi.fn();
  render(
    <GenerationFeed
      {...element(
        [generation({ prompt: "un chat", rewrittenPrompt: "A ginger cat at dusk" })],
        handlers(),
      ).props}
      onUsePrompt={onUsePrompt}
    />,
  );
  await userEvent.click(screen.getByText(/Enhanced prompt/i));
  await userEvent.click(screen.getByRole("button", { name: /use this prompt/i }));
  // Editing and pinning a rewrite need no mechanism of their own: a pinned
  // rewrite is a typed prompt.
  expect(onUsePrompt).toHaveBeenCalledWith("A ginger cat at dusk");
});
