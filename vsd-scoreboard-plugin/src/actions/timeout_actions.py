"""
Timeout management buttons.
Water polo: each team gets 2 timeouts per game.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image, team_bg, dots,
    WHITE, LGRAY, DARK, YELLOW,
)

_SPENT_BG = (35, 35, 45)
_DIM_TEXT  = (120, 120, 140)
_DIM_DOT   = (80, 80, 90)


class _TimeoutUseAction(ScoreboardAction):
    _team: str

    def _on_state_change(self):
        s = self.state
        remaining = s[f"{self._team}_timeouts"]
        active = remaining > 0
        gradient = (team_bg(s, self._team), DARK) if active else (_SPENT_BG, DARK)
        self.set_image(make_button_image(
            bg=None,
            gradient=gradient,
            lines=[
                ("T/O",                        18, WHITE if active else LGRAY),
                (s[f"{self._team}_team"][:6],   9, LGRAY),
                (dots(3 - remaining),          11, YELLOW if active else _DIM_DOT),
            ],
        ))

    def on_key_up(self):
        used = getattr(ScoreboardBridge.get(), f"use_{self._team}_timeout")()
        if used:
            self.show_ok()
        else:
            self.show_alert()


class _TimeoutRestoreAction(ScoreboardAction):
    _team: str

    def _on_state_change(self):
        s = self.state
        remaining = s[f"{self._team}_timeouts"]
        self.set_image(make_button_image(
            bg=_SPENT_BG,
            lines=[
                ("+T/O",                       14, LGRAY),
                (s[f"{self._team}_team"][:6],   9, _DIM_TEXT),
                (dots(3 - remaining),          11, _DIM_DOT),
            ],
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), f"restore_{self._team}_timeout")()
        self.show_ok()


class HometimeoutAction(_TimeoutUseAction):         _team = "home"
class HomerestoretimeoutAction(_TimeoutRestoreAction): _team = "home"
class AwaytimeoutAction(_TimeoutUseAction):         _team = "away"
class AwayrestoretimeoutAction(_TimeoutRestoreAction): _team = "away"
