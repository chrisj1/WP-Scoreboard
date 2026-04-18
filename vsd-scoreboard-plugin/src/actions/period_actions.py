"""
Period and clock control buttons.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image,
    WHITE, LGRAY, DARK, GREEN, YELLOW, ORANGE, TEAL,
)

_PERIOD_BG  = (40, 40, 60)
_PERIOD_DIM = (20, 20, 40)


def _period_label(s: dict) -> str:
    return s["period_text"] or f"Q{s['period']}"


# ── Period navigation ──────────────────────────────────────────────────────────

class _PeriodNavAction(ScoreboardAction):
    _icon:       str    # button label, e.g. "▶ NEXT"
    _gradient:   tuple  # (top_rgb, bottom_rgb)
    _text_color: tuple  # period label text color
    _method:     str    # bridge method name

    def _on_state_change(self):
        s = self.state
        self.set_image(make_button_image(
            bg=None,
            gradient=self._gradient,
            lines=[
                (self._icon,        14, WHITE),
                ("PERIOD",           9, LGRAY),
                (_period_label(s),  10, self._text_color),
            ],
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), self._method)()
        self.show_ok()


class NextperiodAction(_PeriodNavAction):
    _icon = "▶ NEXT"; _gradient = (_PERIOD_BG, _PERIOD_DIM)
    _text_color = YELLOW; _method = "next_period"

class PrevperiodAction(_PeriodNavAction):
    _icon = "◀ PREV"; _gradient = (_PERIOD_DIM, DARK)
    _text_color = LGRAY; _method = "prev_period"


# ── Period text setters ────────────────────────────────────────────────────────

class _SetPeriodTextAction(ScoreboardAction):
    _target:     str    # period_text value to match, e.g. "Final"
    _active_bg:  tuple
    _main_lines: list   # list of (text, size) — drawn in WHITE

    def _on_state_change(self):
        active = self.state["period_text"] == self._target
        self.set_image(make_button_image(
            bg=self._active_bg if active else _PERIOD_DIM,
            lines=[(t, sz, WHITE) for t, sz in self._main_lines] + [("set", 9, LGRAY)],
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), f"set_{self._target.lower()}")()
        self.show_ok()


class SetfinalAction(_SetPeriodTextAction):
    _target    = "Final"
    _active_bg = (GREEN[0] - 40, GREEN[1] - 40, GREEN[2] - 20)
    _main_lines = [("FINAL", 16)]

class SetshootoutAction(_SetPeriodTextAction):
    _target    = "Shootout"
    _active_bg = (ORANGE[0] - 30, ORANGE[1] - 60, 0)
    _main_lines = [("SHOOT", 14), ("OUT", 14)]


# ── Shot clock ─────────────────────────────────────────────────────────────────

class ResetshotclockAction(ScoreboardAction):
    """Reset shot clock to 30 seconds."""

    def _on_state_change(self):
        s = self.state
        self.set_image(make_button_image(
            bg=None,
            gradient=(TEAL, DARK),
            lines=[
                ("RESET",             12, WHITE),
                ("SHOT",              10, LGRAY),
                (str(s["shot_clock"]), 16, WHITE),
            ],
        ))

    def on_key_up(self):
        ScoreboardBridge.get().reset_shot_clock()
        self.show_ok()


class Shotclock35Action(ScoreboardAction):
    """Set shot clock to 35 seconds (after a goal or out of bounds)."""

    def _on_state_change(self):
        self.set_image(make_button_image(
            bg=None,
            gradient=(TEAL, DARK),
            lines=[
                ("35",   26, WHITE),
                ("SHOT",  9, LGRAY),
            ],
        ))

    def on_key_up(self):
        ScoreboardBridge.get().set_shot_clock(35)
        self.show_ok()
