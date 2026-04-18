import threading
import time
from src.core.logger import Logger


class Timer:
    """Manages recurring callbacks identified by string IDs."""

    def __init__(self):
        self._intervals = {}   # id -> {"delay": ms, "callback": fn, "last": float}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_interval(self, interval_id: str, delay_ms: int, callback):
        """Register or update a recurring callback (delay in milliseconds)."""
        with self._lock:
            self._intervals[interval_id] = {
                "delay": delay_ms / 1000.0,
                "callback": callback,
                "last": time.monotonic(),
            }

    def clear_interval(self, interval_id: str):
        """Remove a recurring callback."""
        with self._lock:
            self._intervals.pop(interval_id, None)

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            now = time.monotonic()
            with self._lock:
                due = [
                    (id_, entry)
                    for id_, entry in self._intervals.items()
                    if now - entry["last"] >= entry["delay"]
                ]
            for id_, entry in due:
                try:
                    entry["callback"]()
                except Exception as e:
                    Logger.error(f"Timer callback error [{id_}]: {e}")
                with self._lock:
                    if id_ in self._intervals:
                        self._intervals[id_]["last"] = now
            time.sleep(0.05)
