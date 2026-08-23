/**
 * What Configuration owns, now that it owns the source of the models too.
 *
 * The token is an account-wide secret and the model directory is an
 * application-wide path: neither is a property of any one model, and both used
 * to be reachable only from a view full of per-model controls. What this file
 * pins is the ownership itself — that the field exists here, that saving it goes
 * to the endpoint that writes where `hf auth login` writes, and that it never
 * travels into `server-config.json` with the rest of the form.
 *
 * The seam is `fetch` (see `test-server.ts`). There is no folder chooser to
 * stub: the web build has none, deliberately, so the paths are typed.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Configuration } from "./panels/Configuration";
import { type FakeServer, fakeServer } from "./test-server";

const onSaved = vi.fn(async () => {});

let server: FakeServer;

function show(config: unknown = { server: { port: 8765 }, storage: {} }, tokenPresent = false) {
  server.on("GET /admin/models", () => ({ models: [], warnings: [] }));
  server.on("GET /admin/config", () => config);
  server.on("PUT /admin/config", () => ({ ok: true, restartRequired: false, issues: [] }));
  server.on("POST /admin/hf-token", () => ({ ok: true }));
  render(
    <Configuration
      config={config}
      effectiveHfHome="/hf"
      defaultCacheDir="/data/cache"
      hfTokenPresent={tokenPresent}
      adminPasswordSet={false}
      lanAddresses={["192.168.1.19"]}
      onSaved={onSaved}
    />,
  );
  return server.requests;
}

beforeEach(() => {
  server = fakeServer();
  onSaved.mockClear();
});
afterEach(() => {
  server.restore();
  vi.restoreAllMocks();
});

describe("Hugging Face", () => {
  it("keeps the token and the model directory in one section", async () => {
    show();
    const token = await screen.findByLabelText("Hugging Face token");
    const directory = screen.getByLabelText(/Hugging Face model directory/i);
    const section = token.closest("fieldset") as HTMLElement;

    expect(section.contains(directory)).toBe(true);
    // The one storage fact the field itself cannot carry.
    expect(within(section).getByText("no token")).toBeTruthy();
  });

  it("separates the token from the directory without splitting the section", async () => {
    // One subject, two settings: they share a fieldset, and a rule between them
    // does the work a second card would have done at the cost of the
    // relationship.
    show();
    const token = await screen.findByLabelText("Hugging Face token");
    const directory = screen.getByLabelText(/Hugging Face model directory/i);

    const section = token.closest("fieldset") as HTMLElement;
    expect(section).not.toBeNull();
    expect(section.contains(directory)).toBe(true);
    expect(within(section).getByText("Hugging Face")).toBeTruthy();

    // The separator is between them, not around the group.
    const divider = section.querySelector(".setting-divider") as HTMLElement;
    expect(divider).not.toBeNull();
    expect(divider.compareDocumentPosition(token) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
    expect(
      divider.compareDocumentPosition(directory) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("reports a token that is already there without offering to read it", async () => {
    show({ server: {}, storage: {} }, true);
    expect(await screen.findByText("token present")).toBeTruthy();
    // A password field, and empty: the stored value is never fetched back into
    // the interface to be saved again.
    const field = screen.getByLabelText("Hugging Face token") as HTMLInputElement;
    expect(field.type).toBe("password");
    expect(field.value).toBe("");
  });

  it("saves the token through its own endpoint, never into the configuration", async () => {
    const seen = show();

    await userEvent.type(await screen.findByLabelText("Hugging Face token"), "hf_secret");
    await userEvent.click(screen.getByRole("button", { name: "Save token" }));

    await waitFor(() =>
      expect(seen.some((call) => call.path === "/admin/hf-token")).toBe(true),
    );
    const written = seen.find((call) => call.path === "/admin/hf-token")!;
    expect(written.method).toBe("POST");
    expect((written.body as { token: string }).token).toBe("hf_secret");
    // Not written to `server-config.json`, and not by the form's Save either.
    expect(seen.some((call) => call.method === "PUT" && call.path === "/admin/config")).toBe(false);
    // The panel above it is re-read, so "no token" stops contradicting the save.
    expect(onSaved).toHaveBeenCalled();
    // And the field is cleared rather than left holding a secret.
    expect((screen.getByLabelText("Hugging Face token") as HTMLInputElement).value).toBe("");
  });

  it("refuses to save nothing", async () => {
    show();
    const save = (await screen.findByRole("button", { name: "Save token" })) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
  });

  it("saves the rest of the form without touching the token", async () => {
    const seen = show({ server: { port: 8765 }, storage: {}, default_size: "1024x1024" });

    await userEvent.click(await screen.findByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(seen.some((call) => call.method === "PUT" && call.path === "/admin/config")).toBe(true),
    );

    const written = seen.find((call) => call.method === "PUT" && call.path === "/admin/config")!
      .body as Record<string, unknown>;
    expect(Object.keys(written)).not.toContain("hf_token");
    expect(seen.some((call) => call.path === "/admin/hf-token")).toBe(false);
  });
});

describe("Storage", () => {
  it("offers the pre-quantized cache in its own section, with the real default", async () => {
    // A separate section from Hugging Face: downloaded weights and generated
    // copies are different kinds of file, and either can live on its own disk.
    show();
    const field = (await screen.findByLabelText(/Pre-quantized model cache/i)) as HTMLInputElement;

    const section = field.closest("fieldset") as HTMLElement;
    expect(within(section).getByText("Storage")).toBeTruthy();
    // Not the Hugging Face section.
    expect(within(section).queryByLabelText("Hugging Face token")).toBeNull();

    // Nothing configured: the placeholder is the derived default the backend
    // published, rather than a description of where it might be.
    expect(field.value).toBe("");
    expect(field.placeholder).toBe("/data/cache");
    // Typed, not chosen: the web build offers no folder chooser, because that
    // would mean an endpoint listing the filesystem to anything that can reach
    // the server. The field is therefore editable, and there is no chooser.
    expect(field.readOnly).toBe(false);
    expect(within(section).queryByRole("button", { name: /Choose Folder/i })).toBeNull();
    // No "Use default" until there is something to revert.
    expect(within(section).queryByRole("button", { name: "Use default" })).toBeNull();
    // And the retired location is named nowhere.
    expect(document.body.textContent).not.toContain("mflux-server");
  });

  it("records a typed folder and offers a way back to the default", async () => {
    const seen = show();

    const section = (
      (await screen.findByLabelText(/Pre-quantized model cache/i)) as HTMLElement
    ).closest("fieldset") as HTMLElement;
    await userEvent.type(
      within(section).getByLabelText(/Pre-quantized model cache/i),
      "/Volumes/Big/qds-cache",
    );

    await waitFor(() =>
      expect(
        (screen.getByLabelText(/Pre-quantized model cache/i) as HTMLInputElement).value,
      ).toBe("/Volumes/Big/qds-cache"),
    );
    // Typing does not write on its own: the form's Save owns that.
    expect(seen.some((call) => call.method === "PUT" && call.path === "/admin/config")).toBe(false);
    expect(within(section).getByRole("button", { name: "Use default" })).toBeTruthy();

    await userEvent.click(within(section).getByRole("button", { name: "Use default" }));
    expect((screen.getByLabelText(/Pre-quantized model cache/i) as HTMLInputElement).value).toBe(
      "",
    );
  });

  it("saves the cache directory into the storage section, beside the model directory", async () => {
    const seen = show({
      server: { port: 8765 },
      storage: { hf_home: "/Volumes/Big/hf", cache_dir: "/Volumes/Big/qds-cache" },
    });

    await userEvent.click(await screen.findByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(seen.some((call) => call.method === "PUT" && call.path === "/admin/config")).toBe(true),
    );

    const written = seen.find((call) => call.method === "PUT" && call.path === "/admin/config")!
      .body as { storage: Record<string, unknown> };
    // Two independent settings, written together and neither derived from the
    // other.
    expect(written.storage.cache_dir).toBe("/Volumes/Big/qds-cache");
    expect(written.storage.hf_home).toBe("/Volumes/Big/hf");
  });

  it("keeps the model directory where it was", async () => {
    // This slice adds a section; it does not move the Hugging Face one.
    show();
    const directory = await screen.findByLabelText(/Hugging Face model directory/i);
    const cache = screen.getByLabelText(/Pre-quantized model cache/i);
    expect(directory.closest("fieldset")).not.toBe(cache.closest("fieldset"));
  });
});

describe("the form as a whole", () => {
  it("still saves the sections it owns", async () => {
    const seen = show({ server: { port: 8765 }, storage: {}, default_size: "1024x1024" });

    await userEvent.click(await screen.findByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(seen.some((call) => call.method === "PUT" && call.path === "/admin/config")).toBe(true),
    );

    const written = seen.find((call) => call.method === "PUT" && call.path === "/admin/config")!
      .body as Record<string, unknown>;
    expect(Object.keys(written)).not.toContain("hf_token");
    expect(seen.some((call) => call.path === "/admin/hf-token")).toBe(false);
  });
});
