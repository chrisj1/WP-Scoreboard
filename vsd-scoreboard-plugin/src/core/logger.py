import logging
import os
import sys
import threading

_lock = threading.Lock()
_logger = None


def _get_logger():
    global _logger
    with _lock:
        if _logger is None:
            _logger = logging.getLogger("WPScoreboard")
            _logger.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            _logger.addHandler(ch)
            if getattr(sys, "frozen", False):
                # PyInstaller: write log next to the executable
                base = os.path.dirname(sys.executable)
            else:
                base = os.path.join(os.path.dirname(__file__), "..", "..")
            log_path = os.path.join(base, "plugin.log")
            try:
                fh = logging.FileHandler(os.path.abspath(log_path))
                fh.setFormatter(fmt)
                _logger.addHandler(fh)
            except Exception:
                pass
    return _logger


class Logger:
    @staticmethod
    def info(msg):    _get_logger().info(msg)
    @staticmethod
    def error(msg):   _get_logger().error(msg)
    @staticmethod
    def warning(msg): _get_logger().warning(msg)
    @staticmethod
    def debug(msg):   _get_logger().debug(msg)
