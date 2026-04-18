"""
Score adjustment buttons.
Each has a team-colored background that reflects the configured home/away color.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image, team_color,
    WHITE, LGRAY, DARK,
)


class _ScoreAction(ScoreboardAction):
    _team:  str
    _delta: int

    def _on_state_change(self):
        s = self.state
        bg, dim = team_color(s, self._team)
        label = "+1" if self._delta > 0 else "−1"
        size  = 26 if self._delta > 0 else 22
        self.set_image(make_button_image(
            bg=None,
            gradient=(bg, dim) if self._delta > 0 else (dim, DARK),
            lines=[
                (label,                        size, WHITE),
                (s[f"{self._team}_team"][:6],     9, LGRAY),
            ],
            badge=str(s[f"{self._team}_score"]),
        ))

    def on_key_up(self):
        bridge = ScoreboardBridge.get()
        if self._delta > 0:
            getattr(bridge, f"increment_{self._team}_score")()
        else:
            getattr(bridge, f"decrement_{self._team}_score")()
        self.show_ok()


class HomescoreupAction(_ScoreAction):   _team = "home"; _delta =  1
class HomescoredownAction(_ScoreAction): _team = "home"; _delta = -1
class AwayscoreupAction(_ScoreAction):   _team = "away"; _delta =  1
class AwayscoredownAction(_ScoreAction): _team = "away"; _delta = -1
