"""
Exclusion (ejection) control buttons.
Water polo: a player is excluded (sin-bin) for 20 seconds.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image, team_bg,
    WHITE, LGRAY, DARK,
)

_CLEAR_BG = (30, 30, 50)
_DIM_TEXT  = (120, 120, 140)


class _ExclusionAddAction(ScoreboardAction):
    _team: str

    def _on_state_change(self):
        s = self.state
        n = s[f"{self._team}_exclusions"]
        self.set_image(make_button_image(
            bg=None,
            gradient=(team_bg(s, self._team), DARK),
            lines=[
                ("+EXCL",                  14, WHITE),
                (s[f"{self._team}_team"][:6], 9, LGRAY),
            ],
            badge=str(n),
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), f"add_{self._team}_exclusion")()
        self.show_ok()


class _ExclusionClrAction(ScoreboardAction):
    _team: str

    def _on_state_change(self):
        s = self.state
        n = s[f"{self._team}_exclusions"]
        self.set_image(make_button_image(
            bg=_CLEAR_BG,
            lines=[
                ("CLR",                        14, LGRAY),
                ("EXCL",                       11, LGRAY),
                (s[f"{self._team}_team"][:6],   9, _DIM_TEXT),
            ],
            badge=str(n) if n else None,
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), f"clear_{self._team}_exclusions")()
        self.show_ok()


class HomexclusionaddAction(_ExclusionAddAction): _team = "home"
class HomexclusionclrAction(_ExclusionClrAction): _team = "home"
class AwayxclusionaddAction(_ExclusionAddAction): _team = "away"
class AwayxclusionclrAction(_ExclusionClrAction): _team = "away"
