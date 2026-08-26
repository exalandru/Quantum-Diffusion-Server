#!/usr/bin/env python3
"""Draw the DMG window's background, from the app icon's own palette.

The disk image shipped no window settings at all, so Finder opened it with its
defaults: two icons parked top-left in whatever order the filesystem returned
them, which put **Applications to the left of QDS.app** — a drag that reads
right-to-left, backwards from every installer people have used. Fixing the
layout needs a `.DS_Store`; giving the window a face needs a picture, and this
draws it.

The subject is the icon's own gesture, unrolled: scattered particles on the left
resolving band by band into crisp light on the right. Every colour below is read
out of `assets/icons/icon.svg` rather than sampled by eye, so
the window and the icon sitting on it cannot drift apart.

## The output is committed, and that is the point

This script is the source; `assets/dmg/background*.png` are its output and are
checked in. `make build-dmg` uses the committed files and never runs this — so
packaging needs no Python environment, no Pillow, and produces byte-identical
images on any machine. Regenerating is a deliberate, rare act:

    make dmg-background

That split also contains the one thing here that is *not* reproducible. The
wordmark is real text, so it depends on the build machine's fonts — the very
reason `pack-app-icon.mjs` documents for the app icon. Committing
the render pins it once instead of re-rolling that dice on every build.

## What is deliberately absent

**The version.** It would have to be redrawn every release, and a stale one is
worse than none. The volume is already named `QDS <version>`, so Finder puts the
version in the title bar and the sidebar — where it is derived from the bundle
rather than written down a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Geometry ───────────────────────────────────────────────────────────────
#
# The window's content area, in points. Everything below is expressed in this
# space and rendered at `SUPERSAMPLE` times it, so the strokes and particles get
# their antialiasing from the downscale rather than from hand-rolled coverage
# maths.
WIDTH, HEIGHT = 700, 400
SUPERSAMPLE = 4

# ── Palette, from assets/icons/icon.svg ───────────────────────────────────
INK_TOP = (0x12, 0x10, 0x2A)
INK_MID = (0x0B, 0x08, 0x20)
INK_BOTTOM = (0x15, 0x0F, 0x30)
PINK = (0xF5, 0x9A, 0xE0)
VIOLET = (0x6B, 0x4B, 0xD6)
VIOLET_SOFT = (0x8F, 0x7A, 0xE0)
VIOLET_PALE = (0xB9, 0xA6, 0xF5)
LILAC = (0xE8, 0xDE, 0xF8)

#: The bands, as `(x0, x1, y, width, opacity)`.
#:
#: Each one fades in along *its own* length — transparent at its left end,
#: `LILAC` at its right — which is what makes the field read as resolving rather
#: than as six ruled lines. Uneven lengths and weights on purpose: a regular comb
#: looks like a chart.
#:
#: **They frame the icon row rather than crossing it**, and that is the whole
#: layout decision here. The first version spread six bands evenly down the
#: window; composited with the real icons (`make dmg-preview`) it was plainly
#: wrong — bands ran straight through both glyphs, and the drag arrow at y=189
#: landed inside the band at y=190, so its line vanished and only the chevron
#: survived. A background that has to be *looked past* to find the thing you are
#: meant to drag is decoration working against the interface.
#:
#: So the region the icons occupy — glyphs are 92pt centred at y=185, labels to
#: about y=250 — is left empty, and the field lives above and below it. The
#: gesture still reads left-to-right, which is also the direction of the drag.
CLEAR_LANE = (128, 258)

STROKES = [
    # Above the icons.
    (132, 596, 52, 8, 0.42),
    (150, 648, 84, 11, 0.62),
    (128, 604, 112, 9, 0.50),
    # Below them. Everything meaningful stays above y=370, because the bottom of
    # the window may not belong to us: Finder's **path bar** is a per-user
    # preference (`com.apple.finder ShowPathbar`) that a disk image cannot turn
    # off — AppleScript reports `statusbar visible = false` and the strip appears
    # anyway, because it is a different control. Measured on this machine, where
    # the preference is on: it takes ~28pt off the content area and sliced the
    # wordmark in half. So the artwork is composed for the shorter window and
    # simply has more air at the bottom on machines without it.
    (158, 660, 280, 12, 0.68),
    (138, 604, 306, 9, 0.50),
    (148, 566, 326, 7, 0.34),
]

#: Rows the wordmark occupies, so `_check_clear_lane` can keep bands off it.
WORDMARK_LANE = (334, 366)

#: Rows the path bar can cover. Nothing may be drawn here.
PATHBAR_LANE = (372, HEIGHT)

#: Where the bands come from: unresolved particles, `(x, y, radius, colour)`.
#:
#: Kept out of `CLEAR_LANE` for the same reason the bands are, and pushed left of
#: the QDS icon (which spans x=144..236): a particle behind a glyph is not
#: atmosphere, it is a speck of dirt on the screen.
PARTICLES = [
    # Above the lane.
    (40, 44, 2.5, VIOLET_SOFT), (62, 70, 1.8, VIOLET_SOFT),
    (58, 96, 3.0, VIOLET_SOFT), (86, 62, 2.2, VIOLET_PALE),
    (34, 118, 2.0, VIOLET_SOFT), (96, 104, 1.6, VIOLET_PALE),
    (104, 40, 1.4, VIOLET_PALE), (74, 122, 2.8, VIOLET_SOFT),
    # Below it.
    (52, 292, 2.4, VIOLET_SOFT), (88, 316, 2.0, VIOLET_SOFT),
    (42, 344, 1.7, VIOLET_SOFT), (112, 286, 2.6, VIOLET_PALE),
    (120, 348, 1.9, VIOLET_SOFT), (28, 300, 1.5, VIOLET_SOFT),
    (76, 366, 2.2, VIOLET_PALE),
]

#: The pink bloom the bands resolve into: centre and radii, in points.
GLOW = (546, 200, 230, 150, 0.30)

#: The drag arrow, spanning the gap between the two icons.
#:
#: y=185 is the icons' own centre line, so the arrow points *at* them rather than
#: near them. It sits inside `CLEAR_LANE`, which is what makes it visible at all:
#: at y=189 against a band at y=190 its line was swallowed whole and only the
#: chevron survived — caught by compositing the preview, not by looking at the
#: background on its own.
ARROW_Y = 185
ARROW_X0, ARROW_X1 = 268, 420

WORDMARK = "QUANTUM DIFFUSION SERVER"
#: Tried in order. `SFNS` is the system typeface the rest of the product is set
#: in; the others are fallbacks so a machine without it still renders something
#: rather than failing the build.
FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _linear_gradient(width: int, height: int) -> np.ndarray:
    """The room: a 150° wash from indigo through near-black and back.

    150° in CSS is measured clockwise from "to top", so the axis points down and
    to the right — the same direction the icon's own body gradient runs.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    # Direction vector for 150deg, in screen coordinates (y downwards).
    dx, dy = 0.5, 0.8660254
    projection = x * dx + y * dy
    t = (projection - projection.min()) / (projection.max() - projection.min())

    out = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(
            t, [0.0, 0.55, 1.0],
            [INK_TOP[channel], INK_MID[channel], INK_BOTTOM[channel]],
        )
    return out


def _add_glow(canvas: np.ndarray, scale: int) -> None:
    """The bloom the bands resolve into, added rather than blended.

    Screen-style addition, because this is light: blending would darken the
    bands crossing it, and light falling on light gets brighter.
    """
    cx, cy, rx, ry, strength = GLOW
    height, width = canvas.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt(
        ((x - cx * scale) / (rx * scale)) ** 2 + ((y - cy * scale) / (ry * scale)) ** 2
    )
    falloff = np.clip(1.0 - distance, 0.0, 1.0) * strength
    for channel in range(3):
        canvas[..., channel] += falloff * PINK[channel]


def _stroke_gradient(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour and alpha along a band, as a fraction of its own length.

    Four stops, matching the icon: invisible violet at the left end, then
    softening upwards through `VIOLET_PALE` to near-solid `LILAC` at the right.
    """
    stops = [0.0, 0.45, 0.82, 1.0]
    alphas = [0.0, 0.55, 0.75, 0.90]
    colours = [VIOLET, VIOLET_SOFT, VIOLET_PALE, LILAC]

    alpha = np.interp(t, stops, alphas).astype(np.float32)
    colour = np.stack(
        [np.interp(t, stops, [c[i] for c in colours]) for i in range(3)], axis=-1
    ).astype(np.float32)
    return colour, alpha


def _add_strokes(canvas: np.ndarray, scale: int) -> None:
    for x0, x1, y, weight, opacity in STROKES:
        left, right = x0 * scale, x1 * scale
        centre, radius = y * scale, (weight * scale) / 2.0

        # Only the rows the band can touch: a full-canvas pass per band is
        # forty times the arithmetic for the same picture.
        top = max(0, int(centre - radius) - 1)
        bottom = min(canvas.shape[0], int(centre + radius) + 2)
        if bottom <= top:
            continue

        band_y, band_x = np.mgrid[top:bottom, 0:canvas.shape[1]].astype(np.float32)

        # Round caps: inside the run it is a simple distance from the centre
        # line, and past each end it becomes the distance to that end point.
        dx = np.maximum(np.maximum(left - band_x, band_x - right), 0.0)
        dy = np.abs(band_y - centre)
        distance = np.sqrt(dx * dx + dy * dy)
        coverage = np.clip(radius - distance + 0.5, 0.0, 1.0)
        if not coverage.any():
            continue

        t = np.clip((band_x - left) / max(right - left, 1.0), 0.0, 1.0)
        colour, alpha = _stroke_gradient(t)
        weightmap = (coverage * alpha * opacity)[..., None]
        canvas[top:bottom] = canvas[top:bottom] * (1 - weightmap) + colour * weightmap


def _add_particles(canvas: np.ndarray, scale: int) -> None:
    height, width = canvas.shape[:2]
    for px, py, radius, colour in PARTICLES:
        cx, cy, r = px * scale, py * scale, radius * scale
        top, bottom = max(0, int(cy - r) - 2), min(height, int(cy + r) + 3)
        left, right = max(0, int(cx - r) - 2), min(width, int(cx + r) + 3)
        if bottom <= top or right <= left:
            continue
        y, x = np.mgrid[top:bottom, left:right].astype(np.float32)
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        coverage = np.clip(r - distance + 0.5, 0.0, 1.0) * 0.55
        patch = canvas[top:bottom, left:right]
        canvas[top:bottom, left:right] = (
            patch * (1 - coverage[..., None]) + np.array(colour, np.float32) * coverage[..., None]
        )


def _add_arrow(image: Image.Image, scale: int) -> None:
    """The drag affordance: a fading line into a chevron.

    Drawn with Pillow rather than by hand because it is strokes and a polyline,
    and the supersample is doing the antialiasing either way.
    """
    draw = ImageDraw.Draw(image, "RGBA")
    y = ARROW_Y * scale
    x0, x1 = ARROW_X0 * scale, ARROW_X1 * scale
    thickness = max(1, round(2.5 * scale))

    # Segment by segment, so the line can fade in the way the bands do.
    steps = 64
    for step in range(steps):
        t0 = step / steps
        t1 = (step + 1) / steps
        sx = x0 + (x1 - x0 - 8 * scale) * t0
        ex = x0 + (x1 - x0 - 8 * scale) * t1
        alpha = int(255 * (0.10 + 0.75 * t0))
        draw.line([(sx, y), (ex, y)], fill=(*LILAC, alpha), width=thickness)

    head = 7 * scale
    draw.line(
        [(x1 - head * 2, y - head), (x1, y), (x1 - head * 2, y + head)],
        fill=(*LILAC, 242), width=thickness, joint="curve",
    )


def _add_wordmark(image: Image.Image, scale: int) -> None:
    """The product's name, letterspaced, low and quiet.

    Letterspacing is applied by drawing each glyph in turn: Pillow has no
    tracking control, and the alternative — a tighter, unspaced wordmark — reads
    as a label rather than as a mark.
    """
    path = next((p for p in FONTS if Path(p).exists()), None)
    if path is None:
        print("warning: none of the expected fonts were found; wordmark omitted", file=sys.stderr)
        return

    size = round(15 * scale)
    try:
        font = ImageFont.truetype(path, size)
    except OSError:
        print(f"warning: {path} could not be loaded; wordmark omitted", file=sys.stderr)
        return

    draw = ImageDraw.Draw(image, "RGBA")
    tracking = 0.30 * size

    widths = [draw.textlength(ch, font=font) for ch in WORDMARK]
    total = sum(widths) + tracking * (len(WORDMARK) - 1)

    x = (WIDTH * scale - total) / 2
    y = WORDMARK_LANE[0] * scale
    for ch, advance in zip(WORDMARK, widths):
        draw.text((x, y), ch, font=font, fill=(*LILAC, 158))
        x += advance + tracking


def _check_clear_lane() -> None:
    """Refuse to render artwork that would sit under the icons.

    The defect this prevents was found by eye and would have come back the next
    time a band was nudged: the icons and their labels occupy a horizontal lane,
    and anything drawn inside it competes with the two things the window exists
    to show. Cheap to state, impossible to violate silently.
    """
    top, bottom = CLEAR_LANE
    word_top, word_bottom = WORDMARK_LANE
    path_top, _ = PATHBAR_LANE
    offenders: list[str] = []

    for x0, x1, y, weight, _ in STROKES:
        if top - weight / 2 < y < bottom + weight / 2:
            offenders.append(f"band at y={y} (width {weight}) crosses the icon lane")
        if word_top - weight / 2 < y < word_bottom + weight / 2:
            offenders.append(f"band at y={y} (width {weight}) strikes through the wordmark")
        if y + weight / 2 > path_top:
            offenders.append(f"band at y={y} can be covered by Finder's path bar")

    for x, y, radius, _ in PARTICLES:
        if top - radius < y < bottom + radius:
            offenders.append(f"particle at ({x}, {y}) sits in the icon lane")
        if y + radius > path_top:
            offenders.append(f"particle at ({x}, {y}) can be covered by the path bar")

    if word_bottom > path_top:
        offenders.append("the wordmark extends into the path bar's rows")

    if not top <= ARROW_Y <= bottom:
        offenders.append(f"the drag arrow at y={ARROW_Y} is outside the clear lane")

    if offenders:
        raise SystemExit(
            "the artwork would collide with the icons:\n  "
            + "\n  ".join(offenders)
            + f"\n(clear lane is y={top}..{bottom}; see CLEAR_LANE)"
        )


def render() -> Image.Image:
    """The background at `SUPERSAMPLE` times its final size."""
    _check_clear_lane()
    scale = SUPERSAMPLE
    width, height = WIDTH * scale, HEIGHT * scale

    canvas = _linear_gradient(width, height)
    _add_glow(canvas, scale)
    _add_strokes(canvas, scale)
    _add_particles(canvas, scale)

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    _add_arrow(image, scale)
    _add_wordmark(image, scale)
    return image


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out = root / "assets" / "dmg"
    out.mkdir(parents=True, exist_ok=True)

    master = render()
    # Lanczos down from the 4x master: the strokes' soft ends and the particles'
    # sub-pixel edges are the whole reason the supersample exists.
    #
    # `Image.Resampling.LANCZOS`, not the `Image.LANCZOS` alias: the alias is
    # deprecated and Pillow 12 (this project's pin) no longer declares it, so the
    # short spelling type-checks as an error even where it still runs.
    lanczos = Image.Resampling.LANCZOS
    retina = master.resize((WIDTH * 2, HEIGHT * 2), lanczos)
    standard = master.resize((WIDTH, HEIGHT), lanczos)

    retina.save(out / "background@2x.png", optimize=True)
    standard.save(out / "background.png", optimize=True)

    for name in ("background.png", "background@2x.png"):
        path = out / name
        print(f"    {path.relative_to(root)}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
