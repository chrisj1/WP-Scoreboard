"""
Generate placeholder PNG icons for all actions.
Run this once before building: python generate_icons.py
"""

import os
from PIL import Image, ImageDraw, ImageFont

SIZE = 72

DARK2  = (28,  28,  36)
WHITE  = (255, 255, 255)
LGRAY  = (180, 180, 190)
HOME   = (0,   90, 200)
AWAY   = (180,  80,   0)
TEAL   = (20,  150, 140)
RED    = (200,  30,  30)
YELLOW = (160, 120,   0)
PURPLE = (80,  40,  160)
GREEN  = (30,  140,  50)

ACTIONS = {
    # name:  (bg_color, line1, line2)
    "homescoreup":        (HOME,   "+1",     "HOME"),
    "homescoredown":      (HOME,   "−1",     "HOME"),
    "awayscoreup":        (AWAY,   "+1",     "AWAY"),
    "awayscoredown":      (AWAY,   "−1",     "AWAY"),
    "nextperiod":         (PURPLE, "NEXT",   "PER"),
    "prevperiod":         (PURPLE, "PREV",   "PER"),
    "setfinal":           (GREEN,  "SET",    "FINAL"),
    "setshootout":        (GREEN,  "SET",    "S/O"),
    "resetshotclock":     (TEAL,   "RST",    "SHOT"),
    "shotclock35":        (TEAL,   "35",     "SHOT"),
    "homexclusionadd":    (HOME,   "+EX",    "HOME"),
    "homexclusionclr":    (HOME,   "CLR",    "EXCL"),
    "awayxclusionadd":    (AWAY,   "+EX",    "AWAY"),
    "awayxclusionclr":    (AWAY,   "CLR",    "EXCL"),
    "hometimeout":        (HOME,   "T/O",    "HOME"),
    "homerestoretimeout": (HOME,   "+T/O",   "HOME"),
    "awaytimeout":        (AWAY,   "T/O",    "AWAY"),
    "awayrestoretimeout": (AWAY,   "+T/O",   "AWAY"),
    "homemanup":          (YELLOW, "MAN",    "UP H"),
    "awaymanup":          (YELLOW, "MAN",    "UP A"),
    "scoreinfo":          (DARK2,  "SCR",    "INFO"),
    "periodclockinfo":    (DARK2,  "CLK",    "INFO"),
    "shotclockinfo":      (DARK2,  "SHOT",   "INFO"),
    "teamnameinfo":       (DARK2,  "TEAM",   "INFO"),
    "exclusioninfo":      (DARK2,  "EXCL",   "INFO"),
    "timeoutinfo":        (DARK2,  "T/O",    "INFO"),
    "resetgame":          (RED,    "RST",    "GAME"),
    "triggerreplay":      ((20, 60, 100), "INST", "REPLAY"),
}

PLUGIN_ICON_BG = (0, 90, 200)


def _font(size):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_icon(bg, line1, line2, size=SIZE):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=10, fill=(*bg, 255))

    f1 = _font(22)
    f2 = _font(12)

    b1 = draw.textbbox((0, 0), line1, font=f1)
    b2 = draw.textbbox((0, 0), line2, font=f2)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    gap = 4
    total = h1 + gap + h2
    y = (size - total) // 2

    draw.text(((size - w1) // 2, y),        line1, font=f1, fill=(255, 255, 255, 255))
    draw.text(((size - w2) // 2, y + h1 + gap), line2, font=f2, fill=(200, 200, 210, 255))
    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(script_dir, "com.wps.scoreboard.sdPlugin")
    actions_dir = os.path.join(plugin_dir, "images", "actions")
    images_dir  = os.path.join(plugin_dir, "images")
    os.makedirs(actions_dir, exist_ok=True)

    # Plugin-level icon
    plugin_icon = make_icon(PLUGIN_ICON_BG, "WP", "SCORE", size=128)
    plugin_icon.save(os.path.join(images_dir, "plugin-icon.png"))
    print("Generated: images/plugin-icon.png")

    # Action icons
    for name, (bg, l1, l2) in ACTIONS.items():
        img = make_icon(bg, l1, l2)
        path = os.path.join(actions_dir, f"{name}.png")
        img.save(path)
        print(f"Generated: images/actions/{name}.png")

    print(f"\nDone — {len(ACTIONS) + 1} icons written to {plugin_dir}")


if __name__ == "__main__":
    main()
