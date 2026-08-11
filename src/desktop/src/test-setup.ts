// Tauri's `invoke` and `listen` are the only ways these components reach outside
// the webview, so they are the single seam the tests replace. Everything else —
// availability rules, quantization choices, provenance — is exercised as real
// backend-shaped data, never re-invented in the frontend.
import { vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-shell", () => ({ open: vi.fn(async () => {}) }));
