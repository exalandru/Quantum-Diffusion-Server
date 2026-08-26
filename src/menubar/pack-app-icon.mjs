/**
 * Pack `assets/icons/png/*.png` into `assets/icons/icon.icns`.
 *
 * ## Scope: this script does NOT rasterize
 *
 * The PNGs are exported by hand from Illustrator and committed. That is a
 * deliberate narrowing, and the reason is worth recording so nobody re-litigates
 * it: no rasterizer available on macOS can render this artwork correctly at
 * every size the iconset needs. Measured, not assumed:
 *
 *   | renderer  | large sizes            | 16 / 32 / 64 px           |
 *   |-----------|------------------------|---------------------------|
 *   | Chrome    | faithful (0.24 % diff) | impossible — see below    |
 *   | sips      | drops `feDropShadow`   | renders, but not faithful |
 *   | qlmanage  | composites onto white  | same                      |
 *
 * Headless Chrome will not honour a viewport below roughly 256 px: asking for
 * `--window-size=16,16` yields a 16x16 PNG containing zero opaque pixels, and
 * `--force-device-scale-factor` is clamped, so the small slots cannot be
 * produced at all. Rendering large and downscaling defeats the point — the whole
 * reason the artwork exists in three tiers is that a 16 px icon must be a
 * *different drawing*, not a shrunken one, and measuring the downscale against
 * the shipped icon confirmed it: mean absolute difference 18/255 at 16 px.
 *
 * `sips` renders natively at any size and is very close on the flat fills, but
 * it silently discards `feDropShadow` — alpha in the strip below the body reads
 * 13.1 on the shipped icon and 0.0 on the sips render. (Only `icon.svg`
 * carries a shadow; the 16 and 32 pt tiers deliberately have none, so this costs
 * nothing at small sizes and everything at large ones.)
 *
 * The SVGs beside them in `assets/icons/` remain the design source of truth.
 * They are what an
 * Illustrator export should be made to match, and what the previous, resvg-based
 * version of this script rendered before the Tauri app it depended on was
 * removed from the repository — taking `src-tauri/` and its `node_modules` with
 * it and leaving this script dead for months.
 *
 * ## What this script is for
 *
 * `iconutil` neither resizes nor validates: a 256 dropped into
 * `icon_512x512.png` yields an `.icns` that is silently wrong until someone
 * opens Finder in list view. Everything below exists to make that impossible —
 * every file is checked for presence and exact dimensions before packing, and
 * the result is round-tripped back out and re-checked, because that is the only
 * test that sees the `.icns` the way macOS reads it.
 *
 * ## Usage
 *
 *     make app-icon
 *
 * Both `assets/icons/png/*.png` and the resulting `icon.icns` are committed. Run this
 * after re-exporting from Illustrator.
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");
// The artwork lives with the rest of the project's design assets, beside the
// disk image's background, rather than inside the Swift package: none of it is
// a build input to SwiftPM. `bundle-menubar.sh` copies the packed `.icns` into
// QDS.app, and `make-dmg.sh` reuses it as the volume icon.
const source = join(repo, "assets", "icons", "png");
const icons = join(repo, "assets", "icons");
const staging = join(repo, "build", "icon");

/**
 * The `.iconset` contract, in full: [pixels, Apple's name].
 *
 * Every name here is on Apple's whitelist; anything else in the directory (a
 * `.DS_Store` is the classic) makes `iconutil` fail outright, which is why the
 * iconset is staged under `build/` and filled from this table, never a glob.
 *
 * Note the duplicated sizes: `icon_128x128@2x` and `icon_256x256` are both
 * 256 px, and `icon_256x256@2x` and `icon_512x512` are both 512. They may be the
 * same image — macOS picks by slot, not by file — but both slots must exist.
 */
const ICONSET = [
  [16, "icon_16x16.png"],
  [32, "icon_16x16@2x.png"],
  [32, "icon_32x32.png"],
  [64, "icon_32x32@2x.png"],
  [128, "icon_128x128.png"],
  [256, "icon_128x128@2x.png"],
  [256, "icon_256x256.png"],
  [512, "icon_256x256@2x.png"],
  [512, "icon_512x512.png"],
  [1024, "icon_512x512@2x.png"],
];

/** Width, height and colour type straight out of the PNG's IHDR chunk. */
function pngHeader(path) {
  const header = readFileSync(path).subarray(0, 26);
  if (header.subarray(0, 8).toString("latin1") !== "\x89PNG\r\n\x1a\n") {
    throw new Error(`Not a PNG: ${path}`);
  }
  return {
    width: header.readUInt32BE(16),
    height: header.readUInt32BE(20),
    // 6 = RGBA, 4 = grey+alpha, 3 = palette (may carry tRNS), 2 = RGB, 0 = grey.
    colourType: header.readUInt8(25),
  };
}

// ── Collect and check every slot ───────────────────────────────────────────

if (!existsSync(source)) {
  throw new Error(
    `Missing ${source}.\nExport the icon from Illustrator into that directory, ` +
      `one PNG per entry:\n  ` +
      ICONSET.map(([size, name]) => `${name} (${size}x${size})`).join("\n  "),
  );
}

const problems = [];
for (const [size, name] of ICONSET) {
  const file = join(source, name);
  if (!existsSync(file)) {
    problems.push(`${name}: missing (expected ${size}x${size})`);
    continue;
  }
  let header;
  try {
    header = pngHeader(file);
  } catch (error) {
    problems.push(`${name}: ${error.message}`);
    continue;
  }
  if (header.width !== size || header.height !== size) {
    problems.push(`${name}: is ${header.width}x${header.height}, must be ${size}x${size}`);
  }
  // An icon without transparency ships as a square tile with visible corners —
  // the squircle is drawn by the artwork's own alpha, not clipped by macOS. Easy
  // to get wrong in an Illustrator export ("PNG-24, white background").
  if (header.colourType !== 6 && header.colourType !== 4 && header.colourType !== 3) {
    problems.push(`${name}: has no alpha channel (colour type ${header.colourType}); export as PNG-24 with transparency`);
  }
}

if (problems.length) {
  throw new Error(`${problems.length} problem(s) in ${source}:\n  ` + problems.join("\n  "));
}
console.log(`✓ ${ICONSET.length} source PNGs present, correctly sized, with alpha`);

// ── Stage the iconset ──────────────────────────────────────────────────────

const iconset = join(staging, "QDS.iconset");
rmSync(iconset, { recursive: true, force: true });
mkdirSync(iconset, { recursive: true });

for (const [, name] of ICONSET) {
  cpSync(join(source, name), join(iconset, name));
}
console.log(`✓ ${ICONSET.length} entries staged in ${iconset}`);

// ── Pack ───────────────────────────────────────────────────────────────────

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
for (const [size, name] of ICONSET) {
  if (!produced.has(name)) {
    throw new Error(`icon.icns is missing the ${name} representation`);
  }
  const header = pngHeader(join(verify, name));
  if (header.width !== size || header.height !== size) {
    throw new Error(`icon.icns stores ${name} at ${header.width}x${header.height}, expected ${size}x${size}`);
  }
}
console.log(`✓ icon.icns round-trips to all ${ICONSET.length} representations`);
