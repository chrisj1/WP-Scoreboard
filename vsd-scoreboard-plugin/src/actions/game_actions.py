"""
Game-level control buttons (reset, next/prev game).
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image,
    WHITE, LGRAY,
)

_RESET_BG    = (80, 20, 20)
_RESET_TEXT  = (220, 80, 80)
_GAME_NAV_BG = (30, 50, 80)


class ResetgameAction(ScoreboardAction):
    """Reset the entire game back to defaults. Two-step safety: arm on key_down, fire on key_up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._armed = False

    def _on_state_change(self):
        s = self.state
        self.set_image(make_button_image(
            bg=_RESET_BG,
            lines=[
                ("RESET",                      14, _RESET_TEXT),
                ("GAME",                       14, _RESET_TEXT),
                (f"{s['home_score']}–{s['away_score']}", 9, LGRAY),
            ],
        ))

    def on_key_down(self):
        self._armed = True

    def on_key_up(self):
        if self._armed:
            ScoreboardBridge.get().reset_game()
            self._armed = False
            self.show_ok()


class _GameNavAction(ScoreboardAction):
    _label:  str   # top label line ("NEXT" or "PREV")
    _method: str   # bridge method name ("next_game" or "prev_game")

    def _on_state_change(self):
        s = self.state
        home = s.get("home_team", "")
        away = s.get("away_team", "")
        self.set_image(make_button_image(
            bg=_GAME_NAV_BG,
            lines=[
                (self._label,              14, WHITE),
                ("GAME",                   14, WHITE),
                (f"{home[:4]}>{away[:4]}", 8,  LGRAY),
            ],
        ))

    def on_key_up(self):
        getattr(ScoreboardBridge.get(), self._method)()
        self.show_ok()


class NextgameAction(_GameNavAction): _label = "NEXT"; _method = "next_game"
class PrevgameAction(_GameNavAction): _label = "PREV"; _method = "prev_game"
