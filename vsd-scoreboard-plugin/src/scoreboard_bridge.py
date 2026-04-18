"""
ScoreboardBridge
────────────────
Singleton that maintains a WebSocket connection to the OBS scoreboard
plugin (ws://localhost:8766) and holds the authoritative local state.

All action classes call bridge methods (increment_home_score, etc.) instead
of talking to OBS directly.  The bridge serialises every change and pushes
the full state to OBS so nothing ever falls out of sync.
"""

import json
import threading
import time
import websocket
from src.core.logger import Logger

_DEFAULT_STATE = {
    "home_score":         0,
    "away_score":         0,
    "period":             1,
    "period_text":        "",
    "game_clock_minutes": 8,
    "game_clock_seconds": 0,
    "shot_clock":         30,
    "home_team":          "HOME",
    "away_team":          "AWAY",
    "home_exclusions":    0,
    "away_exclusions":    0,
    "home_timeouts":      3,
    "away_timeouts":      3,
    "home_manup":         False,
    "away_manup":         False,
}

_PERIOD_SEQUENCE = [
    (1, ""),
    (2, ""),
    (3, ""),
    (4, ""),
    (5, "OT"),
    (5, "Final"),
]


class ScoreboardBridge:
    _instance = None
    _class_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ScoreboardBridge":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = ScoreboardBridge()
            return cls._instance

    def __init__(self):
        self._state = _DEFAULT_STATE.copy()
        self._state_lock = threading.Lock()
        self._observers: list = []   # callables notified on every state change
        self._ws = None
        self._ws_lock = threading.Lock()
        self._last_local_update = 0.0  # time.monotonic() of last _update() call
        self._connect()
        self._start_poll()

    # ── Connection ─────────────────────────────────────────────────

    def _connect(self):
        def run():
            while True:
                try:
                    Logger.info("ScoreboardBridge: connecting to ws://localhost:8766")
                    app = websocket.WebSocketApp(
                        "ws://localhost:8766",
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )
                    with self._ws_lock:
                        self._ws = app
                    app.run_forever()
                except Exception as e:
                    Logger.error(f"ScoreboardBridge connection error: {e}")
                Logger.info("ScoreboardBridge: retrying in 3 s")
                time.sleep(3)

        threading.Thread(target=run, daemon=True).start()

    def _start_poll(self):
        """Poll OBS for state every second so clock displays stay in sync."""
        def run():
            while True:
                time.sleep(0.25)
                self._request_state()
        threading.Thread(target=run, daemon=True).start()

    def _request_state(self):
        with self._ws_lock:
            ws = self._ws
        if ws and ws.sock:
            try:
                ws.send(json.dumps({"type": "get_state"}))
            except Exception:
                pass

    def _on_open(self, ws):
        Logger.info("ScoreboardBridge: connected – requesting state")
        ws.send(json.dumps({"type": "get_state"}))

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if msg.get("type") == "state":
            # Ignore polled state for 1 s after a local button press so a
            # stale get_state response cannot revert our optimistic update.
            if time.monotonic() - self._last_local_update < 1.0:
                return
            with self._state_lock:
                self._state.update(msg.get("data", {}))
            self._notify()

    def _on_error(self, ws, error):
        Logger.error(f"ScoreboardBridge WS error: {error}")

    def _on_close(self, ws, code, msg):
        Logger.info("ScoreboardBridge: disconnected")

    # ── Observer pattern ───────────────────────────────────────────

    def add_observer(self, cb):
        if cb not in self._observers:
            self._observers.append(cb)

    def remove_observer(self, cb):
        self._observers = [o for o in self._observers if o is not cb]

    def _notify(self):
        for cb in list(self._observers):
            try:
                cb()
            except Exception as e:
                Logger.error(f"ScoreboardBridge observer error: {e}")

    # ── State access ───────────────────────────────────────────────

    @property
    def state(self) -> dict:
        with self._state_lock:
            return self._state.copy()

    # ── Internal update helper ─────────────────────────────────────

    def _update(self, changes: dict):
        self._last_local_update = time.monotonic()
        with self._state_lock:
            self._state.update(changes)
            payload = self._state.copy()
        self._notify()
        with self._ws_lock:
            ws = self._ws
        if ws and ws.sock:
            try:
                ws.send(json.dumps({"type": "update", "data": payload}))
            except Exception as e:
                Logger.error(f"ScoreboardBridge send error: {e}")

    # ── Score ──────────────────────────────────────────────────────

    def increment_home_score(self):
        self._update({"home_score": max(0, self._state["home_score"] + 1)})

    def decrement_home_score(self):
        self._update({"home_score": max(0, self._state["home_score"] - 1)})

    def increment_away_score(self):
        self._update({"away_score": max(0, self._state["away_score"] + 1)})

    def decrement_away_score(self):
        self._update({"away_score": max(0, self._state["away_score"] - 1)})

    # ── Period ─────────────────────────────────────────────────────

    def next_period(self):
        s = self._state
        cur = (s["period"], s["period_text"])
        try:
            idx = _PERIOD_SEQUENCE.index(cur)
            nxt = _PERIOD_SEQUENCE[min(idx + 1, len(_PERIOD_SEQUENCE) - 1)]
        except ValueError:
            nxt = _PERIOD_SEQUENCE[min(s["period"], len(_PERIOD_SEQUENCE) - 1)]
        self._update({"period": nxt[0], "period_text": nxt[1],
                      "shot_clock": 30, "game_clock_minutes": 8, "game_clock_seconds": 0})

    def prev_period(self):
        s = self._state
        cur = (s["period"], s["period_text"])
        try:
            idx = _PERIOD_SEQUENCE.index(cur)
            prv = _PERIOD_SEQUENCE[max(idx - 1, 0)]
        except ValueError:
            prv = _PERIOD_SEQUENCE[max(s["period"] - 2, 0)]
        self._update({"period": prv[0], "period_text": prv[1]})

    def set_final(self):
        self._update({"period": 5, "period_text": "Final"})

    def set_shootout(self):
        self._update({"period": 5, "period_text": "Shootout"})

    # ── Shot clock ─────────────────────────────────────────────────

    def reset_shot_clock(self):
        self._update({"shot_clock": 30})

    def set_shot_clock(self, seconds: int):
        self._update({"shot_clock": max(0, seconds)})

    # ── Game clock ─────────────────────────────────────────────────

    def set_game_clock(self, minutes: int, seconds: int):
        self._update({"game_clock_minutes": minutes, "game_clock_seconds": seconds})

    # ── Exclusions ─────────────────────────────────────────────────

    def add_home_exclusion(self):
        self._update({"home_exclusions": self._state["home_exclusions"] + 1})

    def clear_home_exclusions(self):
        self._update({"home_exclusions": 0})

    def add_away_exclusion(self):
        self._update({"away_exclusions": self._state["away_exclusions"] + 1})

    def clear_away_exclusions(self):
        self._update({"away_exclusions": 0})

    # ── Timeouts ───────────────────────────────────────────────────

    def use_home_timeout(self) -> bool:
        if self._state["home_timeouts"] > 0:
            self._update({"home_timeouts": self._state["home_timeouts"] - 1})
            return True
        return False

    def restore_home_timeout(self):
        self._update({"home_timeouts": min(3, self._state["home_timeouts"] + 1)})

    def use_away_timeout(self) -> bool:
        if self._state["away_timeouts"] > 0:
            self._update({"away_timeouts": self._state["away_timeouts"] - 1})
            return True
        return False

    def restore_away_timeout(self):
        self._update({"away_timeouts": min(3, self._state["away_timeouts"] + 1)})

    # ── Man-up ─────────────────────────────────────────────────────

    def toggle_home_manup(self):
        self._update({"home_manup": not self._state["home_manup"]})

    def toggle_away_manup(self):
        self._update({"away_manup": not self._state["away_manup"]})

    # ── Game selection ─────────────────────────────────────────────

    def next_game(self):
        self._send_game_nav("next_game")

    def prev_game(self):
        self._send_game_nav("prev_game")

    def _send_game_nav(self, msg_type: str):
        with self._ws_lock:
            ws = self._ws
        if not (ws and ws.sock):
            return
        try:
            ws.send(json.dumps({"type": msg_type}))
        except Exception as e:
            Logger.error(f"ScoreboardBridge {msg_type} error: {e}")

    # ── Replay ─────────────────────────────────────────────────────

    def trigger_replay(self):
        """Snapshot the replay buffer right now. Switch to replay scene to play it."""
        with self._ws_lock:
            ws = self._ws
        if not (ws and ws.sock):
            Logger.error("ScoreboardBridge: not connected, cannot trigger replay")
            return
        try:
            ws.send(json.dumps({"type": "trigger_replay"}))
            Logger.info("ScoreboardBridge: trigger_replay sent")
        except Exception as e:
            Logger.error(f"ScoreboardBridge trigger_replay error: {e}")
            return
        # Request updated state after a short delay so new team/color data arrives.
        def _refresh():
            time.sleep(0.3)
            with self._ws_lock:
                ws2 = self._ws
            if ws2 and ws2.sock:
                try:
                    ws2.send(json.dumps({"type": "get_state"}))
                except Exception:
                    pass
        threading.Thread(target=_refresh, daemon=True).start()

    # ── Full reset ─────────────────────────────────────────────────

    def reset_game(self):
        self._update({
            "home_score": 0, "away_score": 0,
            "period": 1, "period_text": "",
            "game_clock_minutes": 8, "game_clock_seconds": 0,
            "shot_clock": 30,
            "home_exclusions": 0, "away_exclusions": 0,
            "home_timeouts": 3, "away_timeouts": 3,
            "home_manup": False, "away_manup": False,
        })
