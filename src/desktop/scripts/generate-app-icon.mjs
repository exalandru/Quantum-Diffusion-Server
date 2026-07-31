/**
 * Build the macOS icon set from the vector artwork.
 *
 * Before this script the ten `icon-mac-*.png` and `icon.icns` were hand-exported
 * from Affinity Designer (its XMP history was still in `icon.png`) and could not
 * be reproduced: no source artwork in the repo, and no written procedure.
 * Changing anything meant owning that one application. Now
 * `src-tauri/icons/artwork/*.svg` is the source and this is the procedure.
 *
 * Rasterizing goes through the Tauri CLI, which is already a devDependency and
 * embeds resvg. `--png` is what makes it usable here: with it the CLI generates
 * *only* the requested sizes, instead of also writing the .ico, the Windows
 * Square*Logo set and the Android/iOS trees into `icons/`. The CLI renders each
 * size from the SVG rather than downscaling one master, which is what a separate
 * resvg dependency would have bought us, in the same engine.
 *
 * Packing is left to `iconutil` rather than to the CLI's default mode, because
 * that mode cannot be reached without the assets we do not want — and because
 * splitting the two steps is what allows a *different SVG per size tier*.
 *
 * ## Why three tiers
 *
 * A dense particle field cannot be shrunk to 16 px. A particle of radius 3 on a
 * 1024 canvas is 0.047 px at 1/64 scale: the render is a near-uniform wash with
 * random bright specks where two particles share a pixel. Apple's own naming
 * anticipates this — `icon_16x16@2x.png` and `icon_32x32.png` are both 32 px but
 * they are not the same drawing, one is a 16 pt icon on Retina and the other a
 * 32 pt icon at 1x. Under a single master they come out byte-identical, which
 * throws away the only lever available. So each tier gets its own artwork, whose
 * rule is to *remove* sub-pixel geometry rather than scale it down.
 *
 * The reduced tiers are optional: absent, the master is used and the fallback is
 * reported. That way the master can be judged at 16 px before deciding.
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const repo = resolve(desktop, "../..");
const artwork = join(desktop, "src-tauri", "icons", "artwork");
const icons = join(desktop, "src-tauri", "icons");
const staging = join(repo, "build", "desktop", "icon");
const tauriCli = join(desktop, "node_modules", "@tauri-apps", "cli", "tauri.js");

const probe = process.argv.includes("--probe");
const keep = process.argv.includes("--keep");

/** Detail tiers, coarsest first. `svg` falls back to the master when missing. */
const TIERS = [
  { name: "16", svg: "icon-16.svg", sizes: [16, 32] },
  { name: "32", svg: "icon-32.svg", sizes: [32, 64] },
  { name: "full", svg: "icon.svg", sizes: [128, 256, 512, 1024] },
];

/**
 * The `.iconset` contract, in full: [tier, pixels, Apple name, repo name].
 *
 * `iconutil` neither resizes nor checks that a file's pixels match its name — a
 * 256 dropped into `icon_512x512.png` yields an `.icns` that is silently wrong
 * until someone opens Finder in list view. Hence the dimension assertion below.
 * Every name here is on Apple's whitelist; anything else in the directory (a
 * `.DS_Store` is the classic) makes `iconutil` fail outright, which is why the
 * iconset is staged under `build/` and filled from this table, never a glob.
 */
const ICONSET = [
  ["16", 16, "icon_16x16.png", "icon-mac-16x16.png"],
  ["16", 32, "icon_16x16@2x.png", "icon-mac-16x16@2x.png"],
  ["32", 32, "icon_32x32.png", "icon-mac-32x32.png"],
  ["32", 64, "icon_32x32@2x.png", "icon-mac-32x32@2x.png"],
  ["full", 128, "icon_128x128.png", "icon-mac-128x128.png"],
  ["full", 256, "icon_128x128@2x.png", "icon-mac-128x128@2x.png"],
  ["full", 256, "icon_256x256.png", "icon-mac-256x256.png"],
  ["full", 512, "icon_256x256@2x.png", "icon-mac-256x256@2x.png"],
  ["full", 512, "icon_512x512.png", "icon-mac-512x512.png"],
  ["full", 1024, "icon_512x512@2x.png", "icon-mac-512x512@2x.png"],
];

/** Width and height straight out of the PNG's IHDR chunk. */
function pngSize(path) {
  const header = readFileSync(path).subarray(0, 24);
  if (header.subarray(0, 8).toString("latin1") !== "\x89PNG\r\n\x1a\n") {
    throw new Error(`Not a PNG: ${path}`);
  }
  return { width: header.readUInt32BE(16), height: header.readUInt32BE(20) };
}

/**
 * Reject a malformed SVG before the CLI sees it.
 *
 * `tauri icon` calls `usvg::Tree::from_data(...).unwrap()`, so a broken file is a
 * Rust panic rather than a legible error. And a `width`/`height` expressed in
 * percentages makes usvg fall back to its 100x100 `default_size` *silently* —
 * the squareness check still passes and the output is quietly wrong.
 */
function validateSvg(path) {
  const source = readFileSync(path, "utf8");
  // A double hyphen inside a comment is invalid XML, and the panic it triggers
  // reports only `InvalidComment` at row 1 col 1 whatever the real location.
  // Easy to write by accident: CSS custom properties are named with one.
  for (const comment of source.match(/<!--[\s\S]*?-->/g) ?? []) {
    if (comment.slice(4, -3).includes("--")) {
      const line = source.slice(0, source.indexOf(comment)).split("\n").length;
      throw new Error(`${path}:${line}: XML comment contains a double hyphen`);
    }
  }
  const root = source.match(/<svg\b[^>]*>/);
  if (!root) {
    throw new Error(`No <svg> root element in ${path}`);
  }
  const dimension = (name) => {
    const found = root[0].match(new RegExp(`\\b${name}="([^"]+)"`));
    if (!found) {
      throw new Error(`${path}: <svg> has no explicit ${name}`);
    }
    if (!/^\d+(\.\d+)?$/.test(found[1])) {
      throw new Error(`${path}: ${name}="${found[1]}" must be a plain number, not a unit or a percentage`);
    }
    return Number(found[1]);
  };
  const width = dimension("width");
  const height = dimension("height");
  if (width !== height) {
    throw new Error(`${path}: artwork must be square, got ${width}x${height}`);
  }
  if (!/\bviewBox="/.test(root[0])) {
    throw new Error(`${path}: <svg> has no viewBox`);
  }
}

function render(svg, outDir, sizes) {
  execFileSync(process.execPath, [tauriCli, "icon", svg, "-o", outDir, "--png", sizes.join(",")], {
    cwd: desktop,
    stdio: ["ignore", "ignore", "inherit"],
  });
}

// ── Prerequisites ──────────────────────────────────────────────────────────

const master = join(artwork, "icon.svg");
if (!existsSync(master)) {
  throw new Error(`Missing master artwork: ${master}`);
}
if (!existsSync(tauriCli)) {
  throw new Error(`Missing Tauri CLI: ${tauriCli}. Run \`npm install\` first.`);
}

// ── Render every tier ──────────────────────────────────────────────────────

if (!keep) {
  rmSync(staging, { recursive: true, force: true });
}
mkdirSync(staging, { recursive: true });

for (const tier of TIERS) {
  const candidate = join(artwork, tier.svg);
  const source = existsSync(candidate) ? candidate : master;
  validateSvg(source);
  // Each tier renders into its own directory: two of them emit a 32x32.png and
  // they must not overwrite each other — that difference is the point.
  render(source, join(staging, `tier-${tier.name}`), tier.sizes);
  const note = source === master && tier.svg !== "icon.svg" ? ` (no ${tier.svg}, using the master)` : "";
  console.log(`✓ tier ${tier.name}: ${tier.sizes.join(", ")} px rendered${note}`);
}

// ── Assemble the iconset ───────────────────────────────────────────────────

const iconset = join(staging, "QDS.iconset");
rmSync(iconset, { recursive: true, force: true });
mkdirSync(iconset, { recursive: true });

for (const [tier, size, appleName] of ICONSET) {
  const source = join(staging, `tier-${tier}`, `${size}x${size}.png`);
  if (!existsSync(source)) {
    throw new Error(`Tier ${tier} did not produce ${size}x${size}.png`);
  }
  const actual = pngSize(source);
  if (actual.width !== size || actual.height !== size) {
    throw new Error(`${source} is ${actual.width}x${actual.height}, expected ${size}x${size}`);
  }
  cpSync(source, join(iconset, appleName));
}
console.log(`✓ ${ICONSET.length} entries staged in ${iconset}`);

const icns = join(icons, "icon.icns");
execFileSync("iconutil", ["-c", "icns", "--output", icns, iconset], { stdio: "inherit" });
console.log(`✓ icon.icns packed`);

// ── Verify what macOS will actually read back ──────────────────────────────

// Round-tripping through iconutil is the only check that sees the icns the way
// the system does, whichever OSTypes it chose to store the slots under.
const verify = join(staging, "verify.iconset");
rmSync(verify, { recursive: true, force: true });
execFileSync("iconutil", ["-c", "iconset", "--output", verify, icns], { stdio: "inherit" });

const produced = new Set(readdirSync(verify));
for (const [, size, appleName] of ICONSET) {
  if (!produced.has(appleName)) {
    throw new Error(`icon.icns is missing the ${appleName} representation`);
  }
  const actual = pngSize(join(verify, appleName));
  if (actual.width !== size || actual.height !== size) {
    throw new Error(`icon.icns stores ${appleName} at ${actual.width}x${actual.height}, expected ${size}x${size}`);
  }
}
console.log(`✓ icon.icns round-trips to all ${ICONSET.length} representations`);

// ── Publish the PNG set ────────────────────────────────────────────────────

// macOS only ever reads the .icns — the bundler copies it verbatim and points
// CFBundleIconFile at it. These stay because `bundle.icon` lists them and Tauri
// errors on a missing path, and because they are the inspectable proof set.
for (const [, , appleName, repoName] of ICONSET) {
  cpSync(join(iconset, appleName), join(icons, repoName));
}
console.log(`✓ ${ICONSET.length} icon-mac-*.png written to ${icons}`);

// ── Optional: is a native small render better than a downscale? ────────────

if (probe) {
  const dir = join(staging, "probe");
  mkdirSync(dir, { recursive: true });
  render(master, dir, [16, 1024]);
  execFileSync("sips", ["-s", "format", "png", "-z", "16", "16", join(dir, "1024x1024.png"),
    "--out", join(dir, "16-downscaled.png")], { stdio: ["ignore", "ignore", "inherit"] });
  console.log(`✓ probe: compare ${join(dir, "16x16.png")} against ${join(dir, "16-downscaled.png")}`);
}
