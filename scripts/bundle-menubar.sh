#!/bin/bash
#
# Assemble QDS.app around the SwiftPM binary.
#
# SwiftPM produces an executable; macOS needs a bundle for a menubar app to have
# an Info.plist (`LSUIElement`), an icon, and a stable bundle identifier — which
# `SMAppService` needs to register a login item, and which decides where
# Application Support lives.
#
# The bundle carries the two things the server is installed from: a `uv` binary
# and the `qds` wheel. That is what makes a first launch work with no network
# for the package itself and no PyPI dependency at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MENUBAR_DIR="$ROOT/src/menubar"
DIST="$ROOT/dist/app"
APP="$DIST/QDS.app"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/src/server/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || { echo "could not read the version from src/server/pyproject.toml"; exit 1; }

WHEEL="$(ls "$ROOT/dist/server/qds-$VERSION-py3-none-any.whl" 2>/dev/null || true)"
[ -n "$WHEEL" ] || {
  echo "dist/server/qds-$VERSION-py3-none-any.whl is missing. Run 'make build-server' first."
  exit 1
}

# Pinned and verified, not whatever happens to be on the maintainer's PATH. This
# binary is copied into every shipped bundle and runs on the user's machine with
# network access at first launch, so "the build host's uv" is a supply chain.
# aarch64 only, deliberately: MLX requires Apple Silicon, so there is no other
# architecture this app runs on.
UV_VERSION="0.11.32"
UV_SHA256="ed336d0ba49db8ef89b2b41fffa372ce63bd032f22a56f001c265891aec32829"
UV_ARCHIVE="uv-aarch64-apple-darwin.tar.gz"
UV_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ARCHIVE"

UV_CACHE="$ROOT/dist/uv/$UV_VERSION"
mkdir -p "$UV_CACHE"
if [ ! -x "$UV_CACHE/uv" ]; then
  echo "==> fetching uv $UV_VERSION"
  curl -fsSL -o "$UV_CACHE/$UV_ARCHIVE" "$UV_URL"
  echo "$UV_SHA256  $UV_CACHE/$UV_ARCHIVE" | shasum -a 256 -c - >/dev/null || {
    echo "uv $UV_VERSION failed its digest check; refusing to bundle it."
    rm -f "$UV_CACHE/$UV_ARCHIVE"
    exit 1
  }
  tar -xzf "$UV_CACHE/$UV_ARCHIVE" -C "$UV_CACHE" --strip-components 1
  [ -x "$UV_CACHE/uv" ] || { echo "the uv archive did not contain a uv binary."; exit 1; }
fi
UV="$UV_CACHE/uv"

echo "==> building QDS $VERSION"
swift build --package-path "$MENUBAR_DIR" -c release --disable-sandbox

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$MENUBAR_DIR/.build/release/QDS" "$APP/Contents/MacOS/QDS"
cp "$MENUBAR_DIR/Resources/icon.icns" "$APP/Contents/Resources/icon.icns"
cp "$WHEEL" "$APP/Contents/Resources/"
# Copied rather than linked so the bundle is self-contained on a machine that has
# no uv of its own.
cp "$UV" "$APP/Contents/Resources/uv"
chmod +x "$APP/Contents/Resources/uv"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>QDS</string>
  <key>CFBundleDisplayName</key><string>Quantum Diffusion Server</string>
  <key>CFBundleExecutable</key><string>QDS</string>
  <key>CFBundleIdentifier</key><string>com.exalandru.qds</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <!-- A status item, not an app: no Dock icon and no window. -->
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# Ad-hoc, as the Tauri build was. Not notarized: the README tells the user how to
# clear the quarantine attribute, because pretending otherwise would be worse.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || {
  echo "warning: ad-hoc signing failed; the app will still run from a local build"
}

echo "==> $APP"
du -sh "$APP" | awk '{print "    " $1}'
