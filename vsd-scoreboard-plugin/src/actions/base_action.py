"""
Shared helpers for all scoreboard actions:
  - PIL-based 72×72 button image generation
  - Color utilities and palette
  - ScoreboardAction base class (observer lifecycle)
"""

import base64
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from src.core.action import Action
from src.scoreboard_bridge import ScoreboardBridge

SIZE = 72   # VSDinside button canvas size

# ── Font helpers ───────────────────────────────────────────────────────────────

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Image builder ──────────────────────────────────────────────────────────────

def make_button_image(
    bg: tuple,                   # solid background colour (R,G,B) or None for gradient
    lines: list,                 # list of (text, size_pt, colour_rgb)
    gradient: tuple = None,      # (top_rgb, bottom_rgb) – overrides bg
    badge: str = None,           # small top-right badge text
    badge_color: tuple = (255, 200, 0),
) -> str:
    """Return a data-URI PNG suitable for set_image()."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background
    if gradient:
        c1, c2 = gradient
        for y in range(SIZE):
            t = y / SIZE
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (SIZE - 1, y)], fill=(r, g, b, 255))
        # Rounded mask
        mask = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=8, fill=255)
        img.putalpha(mask)
    elif bg:
        draw.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=8,
                                fill=(*bg, 255))

    # Text lines (vertically centred)
    total_h = sum(sz + 2 for _, sz, _ in lines)
    y = (SIZE - total_h) // 2
    for text, size, color in lines:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((SIZE - tw) // 2, y), text, font=font, fill=(*color, 255))
        y += size + 2

    # Optional badge (top-right corner)
    if badge:
        bfont = _load_font(9)
        bbbox = draw.textbbox((0, 0), badge, font=bfont)
        bw = bbbox[2] - bbbox[0] + 6
        draw.rounded_rectangle([SIZE - bw - 2, 2, SIZE - 2, 14], radius=3,
                                fill=(*badge_color, 230))
        draw.text((SIZE - bw + 1, 3), badge, font=bfont, fill=(0, 0, 0, 255))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Colour helpers ─────────────────────────────────────────────────────────────

def argb_to_rgb(argb) -> tuple:
    """Convert an OBS ARGB integer (e.g. 0xFF0080FF) to an (R, G, B) tuple."""
    if not isinstance(argb, int):
        return argb   # already a tuple — pass through
    return ((argb >> 16) & 0xFF, (argb >> 8) & 0xFF, argb & 0xFF)


def darken(rgb: tuple, factor: float = 0.6) -> tuple:
    """Return a darkened version of an RGB tuple."""
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


_HOME_FALLBACK = (0,  90, 200)
_AWAY_FALLBACK = (180, 80,  0)


def team_color(state: dict, team: str) -> tuple:
    """Return (bg, dim) for the given team ('home' or 'away')."""
    fallback = _HOME_FALLBACK if team == "home" else _AWAY_FALLBACK
    bg = argb_to_rgb(state.get(f"{team}_color", fallback))
    return bg, darken(bg)


def team_bg(state: dict, team: str) -> tuple:
    """Return just the background colour for the given team."""
    return team_color(state, team)[0]


def dots(used: int, total: int = 3) -> str:
    """Return a dot indicator string, e.g. ●●○ for 2 used out of 3."""
    return "●" * used + "○" * (total - used)


# ── Colour palette ─────────────────────────────────────────────────────────────

DARK   = (18,  18,  22)
DARK2  = (28,  28,  36)
RED    = (200, 30,  30)
GREEN  = (30,  160, 60)
YELLOW = (220, 180,  0)
ORANGE = (220, 120,  0)
BLUE   = (30,  100, 200)
PURPLE = (120, 40,  180)
TEAL   = (20,  150, 140)
WHITE  = (255, 255, 255)
LGRAY  = (180, 180, 190)


# ── Base for scoreboard actions ────────────────────────────────────────────────

class ScoreboardAction(Action):
    """Subclass this for any action that needs live scoreboard state."""

    def on_will_appear(self):
        ScoreboardBridge.get().add_observer(self._on_state_change)
        self._on_state_change()          # draw immediately with current state

    def on_will_disappear(self):
        ScoreboardBridge.get().remove_observer(self._on_state_change)

    @property
    def state(self) -> dict:
        """Convenience accessor for the current scoreboard state."""
        return ScoreboardBridge.get().state

    def _on_state_change(self):
        """Called whenever bridge state changes. Override to redraw."""
        pass
