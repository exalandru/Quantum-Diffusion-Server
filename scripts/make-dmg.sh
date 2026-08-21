#!/bin/bash
#
# Package QDS.app into a distributable disk image.
#
# The DMG was produced by the Tauri bundler before v2. SwiftPM has no bundler,
# so the app is assembled by hand (`bundle-menubar.sh`) and the image is made
# here: `hdiutil` over a staging directory holding the app and a symlink to
# /Applications, which is the drag-to-install layout macOS users expect.
#
# The version is read from the built app's `Info.plist` rather than from
# `pyproject.toml`. Both would give the same answer today, but only one of them
# is a statement about the bundle actually being packaged -- reading the source
# again would let a stale `dist/app/QDS.app` ship under a fresh version number.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist/app"
APP="$DIST/QDS.app"

[ -d "$APP" ] || {
  echo "$APP is missing. Run 'make build-app' first."
  exit 1
}

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  "$APP/Contents/Info.plist" 2>/dev/null || true)"
[ -n "$VERSION" ] || { echo "could not read CFBundleShortVersionString from the bundle"; exit 1; }

DMG="$DIST/QDS-$VERSION.dmg"
STAGE="$(mktemp -d)"
# Removed whatever happens: hdiutil failing halfway would otherwise leave a full
# copy of the app in the temporary directory.
trap 'rm -rf "$STAGE"' EXIT

echo "==> packaging QDS $VERSION"
# `cp -R` and not a move: `dist/app/QDS.app` stays where `make build-app` put it,
# so building the image twice does not consume the thing it packages.
cp -R "$APP" "$STAGE/QDS.app"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
# UDZO: read-only and compressed, which is what a distributed image should be.
# `-quiet` because hdiutil's progress is noise in a build log; failures still
# print, since only stdout is suppressed.
hdiutil create \
  -volname "QDS $VERSION" \
  -srcfolder "$STAGE" \
  -format UDZO \
  -quiet \
  "$DMG"

# Ad-hoc, exactly as the app inside it is, and for the same reason: this is not
# a notarized release. The README tells the user how to clear the quarantine
# attribute, which is the honest thing to do rather than implying otherwise.
codesign --force --sign - "$DMG" >/dev/null 2>&1 || {
  echo "warning: ad-hoc signing the image failed; it still mounts"
}

echo "==> $DMG"
du -sh "$DMG" | awk '{print "    " $1}'
