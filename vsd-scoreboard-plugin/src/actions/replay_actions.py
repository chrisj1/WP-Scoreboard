"""
Instant Replay button — saves a replay snapshot via the OBS scoreboard plugin.
Press the button to capture the last N seconds. Switch to the replay scene to play it.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image,
    WHITE, LGRAY,
)

_REPLAY_BG = (20, 60, 100)


class TriggerreplayAction(ScoreboardAction):
    """Save a replay snapshot right now."""

    def _on_state_change(self):
        self.set_image(make_button_image(
            bg=_REPLAY_BG,
            lines=[
                ("INSTANT", 12, WHITE),
                ("REPLAY",  14, (80, 180, 255)),
                ("● REC",    9, LGRAY),
            ],
        ))

    def on_key_up(self):
        ScoreboardBridge.get().trigger_replay()
        self.show_ok()
