"""
Read-only info display buttons.
These show live scoreboard data and update automatically.
"""

from src.actions.base_action import (
    ScoreboardAction, ScoreboardBridge,
    make_button_image, dots,
    DARK, DARK2, RED, YELLOW, WHITE, LGRAY,
)


class ScoreinfoAction(ScoreboardAction):
    """Shows HOME score | AWAY score with team-color gradient background."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        home = s["home_score"]
        away = s["away_score"]
        self.set_title("")       # image carries all the info
        self.set_image(make_button_image(
            bg=None,
            gradient=(DARK2, DARK),
            lines=[
                (f"{home}  –  {away}", 22, WHITE),
                (f"{s['home_team'][:4]}  {s['away_team'][:4]}", 9, LGRAY),
            ],
        ))


class PeriodclockinfoAction(ScoreboardAction):
    """Shows period label + game clock."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        period = s["period_text"] or f"Q{s['period']}"
        clock = f"{s['game_clock_minutes']:02d}:{s['game_clock_seconds']:02d}"
        self.set_title("")
        self.set_image(make_button_image(
            bg=None,
            gradient=(DARK2, DARK),
            lines=[
                (period, 20, WHITE),
                (clock,  18, LGRAY),
            ],
        ))


class ShotclockinfoAction(ScoreboardAction):
    """Shows shot clock – turns red background when ≤ 5 seconds."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        sc = s["shot_clock"]
        urgent = sc <= 5
        bg     = (RED[0]-20, RED[1], RED[2]) if urgent else DARK2
        color  = WHITE
        self.set_title("")
        self.set_image(make_button_image(
            bg=bg,
            lines=[
                (str(sc),       28, color),
                ("SHOT CLK",     9, LGRAY),
            ],
        ))


class TeamnameinfoAction(ScoreboardAction):
    """Shows both team names and their current records."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        home_rec = f"{s['home_wins']}-{s['home_losses']}"
        away_rec = f"{s['away_wins']}-{s['away_losses']}"
        self.set_title("")
        self.set_image(make_button_image(
            bg=DARK2,
            lines=[
                (s["home_team"][:8], 11, WHITE),
                (home_rec,            9, LGRAY),
                (s["away_team"][:8], 11, WHITE),
                (away_rec,            9, LGRAY),
            ],
        ))


class ExclusioninfoAction(ScoreboardAction):
    """Shows current exclusion counts for both teams."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        self.set_title("")
        self.set_image(make_button_image(
            bg=DARK2,
            lines=[
                ("EXCLUSIONS",          9, LGRAY),
                (f"H {s['home_exclusions']}  A {s['away_exclusions']}", 18, WHITE),
            ],
        ))


class TimeoutinfoAction(ScoreboardAction):
    """Shows remaining timeouts for both teams."""

    def _on_state_change(self):
        s = ScoreboardBridge.get().state
        dots_h = dots(s["home_timeouts"])
        dots_a = dots(s["away_timeouts"])
        self.set_title("")
        self.set_image(make_button_image(
            bg=DARK2,
            lines=[
                ("TIMEOUTS",  9, LGRAY),
                (f"H {dots_h}", 12, WHITE),
                (f"A {dots_a}", 12, WHITE),
            ],
        ))
