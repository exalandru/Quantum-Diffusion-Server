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
#
# ## Why this is a two-stage build
#
# A compressed image is read-only, and Finder's window settings -- background
# picture, icon size and coordinates, window frame -- live in a `.DS_Store` that
# only Finder can write, only on a *mounted, writable* volume. So: create a UDRW
# image, mount it, let Finder arrange it, unmount, then convert to UDZO. The
# single `hdiutil create -format UDZO` this replaces could not carry any of it,
# which is why the window opened with Finder's defaults -- and why Applications
# sat to the LEFT of QDS.app, making the install drag run right-to-left.
#
# The background is `assets/dmg/background*.png`, committed to the repository and
# regenerated only on demand by `make dmg-background`. Nothing here draws it, so
# packaging needs no Python and no image libraries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist/app"
APP="$DIST/QDS.app"
ART="$ROOT/assets/dmg"

[ -d "$APP" ] || {
  echo "$APP is missing. Run 'make build-app' first."
  exit 1
}

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  "$APP/Contents/Info.plist" 2>/dev/null || true)"
[ -n "$VERSION" ] || { echo "could not read CFBundleShortVersionString from the bundle"; exit 1; }

DMG="$DIST/QDS-$VERSION.dmg"
VOLUME="QDS $VERSION"
STAGE="$(mktemp -d)"
RW="$(mktemp -u)/rw.dmg"
mkdir -p "$(dirname "$RW")"

MOUNTPOINT=""
cleanup() {
  # Detach before deleting: a still-mounted image cannot be removed, and the
  # loop device would leak for the rest of the session.
  if [ -n "$MOUNTPOINT" ] && [ -d "$MOUNTPOINT" ]; then
    hdiutil detach "$MOUNTPOINT" -force -quiet 2>/dev/null || true
  fi
  rm -rf "$STAGE" "$(dirname "$RW")"
}
trap cleanup EXIT

echo "==> packaging QDS $VERSION"
# `cp -R` and not a move: `dist/app/QDS.app` stays where `make build-app` put it,
# so building the image twice does not consume the thing it packages.
cp -R "$APP" "$STAGE/QDS.app"
ln -s /Applications "$STAGE/Applications"

# The window dressing, when the artwork is present. Absent, the image is still
# built and still installs -- it just opens with Finder's defaults, and the build
# says so rather than failing a release over a picture.
DRESSED=0
if [ -f "$ART/background.png" ] && [ -f "$ART/background@2x.png" ]; then
  mkdir -p "$STAGE/.background"
  # One TIFF carrying both resolutions. Finder picks the @2x representation on a
  # Retina display; a lone 1x PNG would be upscaled and look soft on every modern
  # Mac. `tiffutil -cathidpicheck` is what marks the pair as 1x/2x of one image.
  tiffutil -cathidpicheck "$ART/background.png" "$ART/background@2x.png" \
    -out "$STAGE/.background/background.tiff" >/dev/null 2>&1 && DRESSED=1 || {
    echo "warning: could not build the HiDPI background; falling back to the 1x PNG"
    cp "$ART/background.png" "$STAGE/.background/background.png"
    DRESSED=2
  }
else
  echo "warning: $ART/background*.png missing; run 'make dmg-background'"
fi

# The volume's own icon, so the mounted disk is QDS on the desktop and in the
# sidebar rather than a blank drive. The name is fixed by convention: Finder
# looks for `.VolumeIcon.icns` and nothing else.
if [ -f "$ROOT/assets/icons/icon.icns" ]; then
  cp "$ROOT/assets/icons/icon.icns" "$STAGE/.VolumeIcon.icns"
fi

rm -f "$DMG"

# Stage one: a writable image, sized with room for the .DS_Store and the
# background Finder is about to write into it.
hdiutil create \
  -volname "$VOLUME" \
  -srcfolder "$STAGE" \
  -format UDRW \
  -fs HFS+ \
  -quiet \
  "$RW"

MOUNTPOINT="/Volumes/$VOLUME"
# A volume of this name left over from an interrupted build has to go first.
# macOS does not refuse a duplicate name, it *renames* the new mount to
# "QDS 2.2.1 1" — so `tell disk "QDS 2.2.1"` would find the stale one and dress
# it, and this build would ship an undressed image with no error anywhere.
if [ -d "$MOUNTPOINT" ]; then
  echo "    detaching a leftover $VOLUME"
  hdiutil detach "$MOUNTPOINT" -force -quiet 2>/dev/null || {
    echo "$MOUNTPOINT is mounted and could not be detached; unmount it and retry."
    exit 1
  }
fi

# Mounted at its natural place under /Volumes, deliberately, and not at a
# temporary directory via `-mountpoint`. Finder addresses a volume as
# `disk "<name>"`, and it only resolves that for volumes mounted under /Volumes:
# with a custom mountpoint every `tell disk` below fails with -1728 ("can't get
# disk"), the window is never dressed, and the build reports the fallback
# warning while producing a plain image. Measured, not assumed.
#
# `-nobrowse` still keeps it out of the user's sidebar while the build runs;
# it does not hide the volume from Apple Events.
hdiutil attach "$RW" -nobrowse -noautoopen -quiet
[ -d "$MOUNTPOINT" ] || { echo "the image did not mount at $MOUNTPOINT"; exit 1; }

# The volume icon only takes effect once the volume is flagged as having one.
if [ -f "$MOUNTPOINT/.VolumeIcon.icns" ]; then
  SetFile -a C "$MOUNTPOINT" 2>/dev/null || \
    echo "warning: could not set the custom-icon flag; the volume keeps the generic icon"
fi

# Hide the machinery BEFORE Finder looks at the volume.
#
# Ordering is load-bearing and was got wrong first time: run after the
# AppleScript, the `.background` folder had already been catalogued into the
# .DS_Store as a visible item and showed up in the finished image's window,
# top-left, captioned ".background". Hiding it first means Finder never records
# it. (`.DS_Store` and `.VolumeIcon.icns` are dot-files, which Finder hides on
# its own; `.background` is too, but the flag is what keeps it out of the
# arrangement rather than merely out of sight.)
chflags hidden "$MOUNTPOINT/.background" 2>/dev/null || true
chflags hidden "$MOUNTPOINT/.VolumeIcon.icns" 2>/dev/null || true

# Caption the icon "QDS", not "QDS.app".
#
# Set on the MOUNTED copy, before Finder is told to arrange the window: applied
# to the staging directory instead, the flag did not survive `hdiutil create`
# and the finished image still read "QDS.app". `-a E` is the extension-hidden
# bit; it lives in the item's own metadata, so `dist/app/QDS.app` and the
# installed application are untouched either way.
#
# Best effort by nature, and the limit is worth stating so a future reader does
# not chase it: a user who has turned on "Show all filename extensions"
# (`NSGlobalDomain AppleShowAllExtensions`, off by default) sees "QDS.app"
# regardless. That preference deliberately overrides the per-item flag, and no
# disk image can opt out of it — verified on a machine with it enabled, where
# the flag is set and Finder shows the extension anyway.
SetFile -a E "$MOUNTPOINT/QDS.app" 2>/dev/null || \
  echo "warning: could not hide the .app extension; the icon will read 'QDS.app'"

# `bounds` is the window's frame, and the background fills its *content* area —
# the two differ by the title bar. Setting {200,140,900,580} gave a 700-wide
# frame whose content was 700x400 minus the chrome, so Finder tiled the picture
# and left a bare strip down the right-hand side. Measured on this machine, the
# title bar is 28pt tall, so the frame has to be that much taller than the
# artwork. `set bounds` after `open` and re-read below, because a window
# constrained by the screen silently gets a different size than the one asked
# for — and then the picture would not line up again.
WINDOW_LEFT=200
WINDOW_TOP=140
TITLEBAR=28

# Stage two: Finder arranges the window and writes .DS_Store.
#
# Everything below is deliberate:
#
# * QDS.app on the LEFT, Applications on the RIGHT. This is the fix that
#   matters most and it costs nothing: the default layout had them reversed,
#   so the drag people are asked to perform ran right-to-left.
# * Coordinates are the *centres* of the icons and must agree with the artwork's
#   clear lane (see scripts/dmg-background.py, CLEAR_LANE) -- the background is
#   drawn to leave this row empty.
# * `arrangement not arranged` and no `sort` key: snap-to-grid would quietly move
#   the icons off the picture.
# * The toolbar, sidebar and status bar are hidden. They are workspace chrome and
#   this window is a poster with two things to drag.
if [ "$DRESSED" != "0" ]; then
  BACKGROUND_FILE="background.tiff"
  [ "$DRESSED" = "2" ] && BACKGROUND_FILE="background.png"

  # `|| true`: a build host where Finder is unavailable or refuses automation
  # (no Apple Events permission, a headless CI) must still produce an installable
  # image. It just will not be a dressed one, and the warning says so.
  osascript <<APPLESCRIPT >/dev/null 2>&1 || echo "warning: Finder would not dress the window; the image is plain but valid"
    tell application "Finder"
      tell disk "$VOLUME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set sidebar width of container window to 0
        -- {left, top, right, bottom} is the window FRAME. The background fills
        -- the content area, which is shorter by the title bar, so the frame is
        -- 700 x (400 + titlebar). Getting this wrong tiles the picture and
        -- leaves a bare strip on the right.
        set the bounds of container window to {$WINDOW_LEFT, $WINDOW_TOP, $((WINDOW_LEFT + 700)), $((WINDOW_TOP + 400 + TITLEBAR))}
        set options to the icon view options of container window
        set arrangement of options to not arranged
        set icon size of options to 92
        set text size of options to 12
        set background picture of options to file ".background:$BACKGROUND_FILE"
        set position of item "QDS.app" of container window to {190, 185}
        set position of item "Applications" of container window to {510, 185}
        -- Force the write, then give the daemon a moment to flush it. Without
        -- the delay the .DS_Store is sometimes still buffered when we detach and
        -- the whole arrangement is lost.
        close
        open
        update without registering applications
        delay 2
        close
      end tell
    end tell
APPLESCRIPT
fi

# Put the volume icon back, after Finder has finished with the window.
#
# Finder drops `.VolumeIcon.icns` from the volume while it rearranges: the file
# is present in the staging directory and on the freshly mounted writable image,
# and gone from the finished one. Measured by listing all three — the earlier
# versions of this check looked at the staging directory and at the mounted RW
# volume, both of which still held it, which is why the loss went unnoticed.
#
# Rewriting it here, after the AppleScript and before `sync`, is what makes it
# survive into the converted image.
if [ -f "$ROOT/assets/icons/icon.icns" ]; then
  cp "$ROOT/assets/icons/icon.icns" "$MOUNTPOINT/.VolumeIcon.icns"
  chflags hidden "$MOUNTPOINT/.VolumeIcon.icns" 2>/dev/null || true
  # The flag has to be (re)set after the file exists, or it points at nothing.
  SetFile -a C "$MOUNTPOINT" 2>/dev/null || \
    echo "warning: could not set the custom-icon flag; the volume keeps the generic icon"
fi

sync
hdiutil detach "$MOUNTPOINT" -quiet
MOUNTPOINT=""

# Stage three: the read-only, compressed image that actually ships. UDZO is what
# a distributed image should be, and the conversion preserves the .DS_Store.
hdiutil convert "$RW" -format UDZO -imagekey zlib-level=9 -quiet -o "$DMG"

# Ad-hoc, exactly as the app inside it is, and for the same reason: this is not
# a notarized release. The README tells the user how to clear the quarantine
# attribute, which is the honest thing to do rather than implying otherwise.
codesign --force --sign - "$DMG" >/dev/null 2>&1 || {
  echo "warning: ad-hoc signing the image failed; it still mounts"
}

# The volume icon has to survive everything above. It is the one dressing step
# with no visible failure mode: a volume flagged `-a C` whose `.VolumeIcon.icns`
# went missing mounts with the generic white drive icon and reports no error
# anywhere. So the finished image is opened and checked, rather than the staging
# directory or the writable volume — both of those hold the file even in the
# failing case, which is exactly why this went unnoticed until it was measured.
if [ -f "$ROOT/assets/icons/icon.icns" ]; then
  CHECK="$(mktemp -d)"
  if hdiutil attach "$DMG" -mountpoint "$CHECK" -nobrowse -noautoopen -quiet 2>/dev/null; then
    [ -f "$CHECK/.VolumeIcon.icns" ] || {
      echo "warning: .VolumeIcon.icns is missing from the finished image;"
      echo "         the disk will mount with the generic icon"
    }
    hdiutil detach "$CHECK" -force -quiet 2>/dev/null || true
  fi
  rmdir "$CHECK" 2>/dev/null || true
fi

echo "==> $DMG"
du -sh "$DMG" | awk '{print "    " $1}'
