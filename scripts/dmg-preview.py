#!/usr/bin/env python3
"""Preview the DMG window as Finder will actually draw it.

The background alone cannot be judged: what matters is whether the two icons and
their labels stay readable on top of it, and whether the drag arrow lands in a
gap rather than inside a band. This composites the real icon, at the real
coordinates the `.DS_Store` will use, onto the real background.

Not part of any build. `make dmg-preview` writes it to /tmp for inspection.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# The layout, in points. These are the numbers the AppleScript in make-dmg.sh
# sets, kept here so the preview cannot drift from what ships.
WINDOW = (700, 400)
ICON_SIZE = 92
QDS_CENTRE = (190, 185)
APPLICATIONS_CENTRE = (510, 185)
LABEL_OFFSET = 34  # baseline below the glyph's bottom edge

SCALE = 2  # preview at @2x, like a Retina screen


def _icon(path: Path, size: int) -> Image.Image:
    """Rasterise an .icns at `size`, via sips."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "icon.png"
        subprocess.run(
            ["sips", "-s", "format", "png", "--resampleHeightWidth", str(size), str(size),
             str(path), "--out", str(out)],
            check=True, capture_output=True,
        )
        return Image.open(out).convert("RGBA")


def _applications_icon(size: int) -> Image.Image:
    """The Applications folder, as Finder draws it."""
    candidates = [
        Path("/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/ApplicationsFolderIcon.icns"),
        Path("/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericFolderIcon.icns"),
    ]
    for path in candidates:
        if path.exists():
            return _icon(path, size)
    # A blue square is a poor stand-in, but a missing preview is worse.
    placeholder = Image.new("RGBA", (size, size), (60, 140, 220, 255))
    return placeholder


def main() -> int:
    background = ROOT / "assets" / "dmg" / "background@2x.png"
    if not background.exists():
        print(f"{background} is missing. Run 'make dmg-background' first.", file=sys.stderr)
        return 1

    canvas = Image.open(background).convert("RGBA")
    if canvas.size != (WINDOW[0] * 2, WINDOW[1] * 2):
        print(f"warning: background is {canvas.size}, expected {(WINDOW[0] * 2, WINDOW[1] * 2)}")

    glyph = ICON_SIZE * SCALE
    qds = _icon(ROOT / "assets" / "icons" / "icon.icns", glyph)
    applications = _applications_icon(glyph)

    draw = ImageDraw.Draw(canvas, "RGBA")
    font_path = "/System/Library/Fonts/SFNS.ttf"
    font = ImageFont.truetype(font_path, round(12.5 * SCALE)) if Path(font_path).exists() else None

    for image, (cx, cy), label in (
        (qds, QDS_CENTRE, "QDS"),
        (applications, APPLICATIONS_CENTRE, "Applications"),
    ):
        x = round(cx * SCALE - glyph / 2)
        y = round(cy * SCALE - glyph / 2)
        canvas.alpha_composite(image, (x, y))
        if font is None:
            continue
        text_y = y + glyph + (LABEL_OFFSET - ICON_SIZE // 2) * SCALE
        width = draw.textlength(label, font=font)
        # Finder draws a rounded translucent plate behind an icon label on a
        # picture background; without it the preview flatters the design.
        pad = 5 * SCALE
        draw.rounded_rectangle(
            [cx * SCALE - width / 2 - pad, text_y - 2 * SCALE,
             cx * SCALE + width / 2 + pad, text_y + 15 * SCALE],
            radius=4 * SCALE, fill=(0, 0, 0, 90),
        )
        draw.text((cx * SCALE - width / 2, text_y), label, font=font, fill=(242, 244, 248, 255))

    out = Path("/tmp/qds-dmg-preview.png")
    canvas.convert("RGB").save(out)
    print(f"    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
