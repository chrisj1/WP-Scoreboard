"""
Man-up (power play) toggle buttons.
Button lights up yellow when man-up is active.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image,
    WHITE, LGRAY, YELLOW,
)

_ACTIVE_BG   = (160, 120,  0)
_INACTIVE_BG = (35,  35,  45)
_DIM_TEAM    = (100, 100, 110)
_DIM_ARROW   = (60,  60,  70)


class _ManupAction(ScoreboardAction):
    _team: str

    def _on_state_change(self):
        s = self.state
        active = s[f"{self._team}_manup"]
        self.set_image(make_button_image(
            bg=_ACTIVE_BG if active else _INACTIVE_BG,
            lines=[
                ("MAN UP" if active else "man up", 13, WHITE if active else LGRAY),
                (s[f"{self._team}_team"][:6],        9, LGRAY if active else _DIM_TEAM),
                ("▲",                              16, YELLOW if active else _DIM_ARROW),
            ],
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), f"toggle_{self._team}_manup")()
        self.show_ok()


class HomemanupAction(_ManupAction): _team = "home"
class AwaymanupAction(_ManupAction): _team = "away"
